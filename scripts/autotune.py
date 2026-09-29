#!/usr/bin/env python3
"""Settings autotuner: find the best dots.tts config for *this* GPU.

Why this exists
---------------
The knobs that decide whether the 5.5 GB / RTF 0.5 targets are met are coupled
in ways that are not obvious from the config file:

* ``TTS_MAX_GENERATE_LENGTH`` is snapped **up** to the next torch.compile bucket
  by ``DotsTtsModel._resolve_generate_length_bucket`` (``(64, 128, 256, 512)``).
  Under ``TTS_OPTIMIZE=true`` a value of 96 therefore behaves exactly like 128,
  but with a tighter validation limit -- pointless. Only bucket-aligned values
  are worth sweeping while ``OPTIMIZE`` is on.
* The DiT KV cache is allocated as ``(num_steps, layers, heads, tokens, dim)``,
  so every extra sampling step doubles it. ``max_generate_length`` and
  ``num_steps`` must be tuned **together**: the pair that fits at 2 steps OOMs
  at 4.
* ``TTS_MAX_SEQUENCE_LENGTH`` sizes the LLM KV cache independently.
* The three patches in ``app/patches.py`` change the budget for all of the above.

Each *server-side* setting costs a container restart plus a ``torch.compile``
warmup (~3-5 min), while per-request settings cost seconds. The tuner exploits
that split: it explores server-side dimensions with restarts and per-request
dimensions in place.

Method
------
Staged coordinate descent -- not a full factorial, which at 3-5 min per config
would take hours. Start from a baseline, sweep one dimension, keep the winner,
move on. Every candidate is recorded (failures included), so a crash or an
``--resume`` never loses work.

Measurement reuses ``scripts/benchmark_test.py`` itself, so the numbers used to
pick a config come from exactly the code that performs acceptance.

Scoring
-------
RTF depends strongly on output length, because the reference-prompt prefill is
a fixed cost amortised over the generated audio (measured: 3.2 s of audio -> RTF
0.70, 13.4 s -> 0.44). Tuning against a single sentence would therefore favour
settings that look bad for short utterances. Each candidate is scored across
several ``--profiles`` text lengths and ranked on the **worst** RTF.

Artifacts
---------
* ``data/output/tuning_report.json``  -- the full matrix, failures included.
* ``data/output/best_config.json``     -- the winning profile, read by app/config.py.
* ``data/output/tuned.env``           -- the same, as a pasteable .env block.
* ``--apply`` writes them into ``.env`` behind a timestamped backup.

Examples
--------
    # full sweep, balanced objective, dry run (default)
    python scripts/autotune.py

    # fast, and actually install the winner
    python scripts/autotune.py --quick --apply
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_PATH = PROJECT_ROOT / ".env"

# ``data/`` is bind-mounted at the same path inside the container (see the
# `volumes:` block in docker-compose.yml), so a report written by the in-container
# benchmark has to be addressed from the other side of that mount.
DATA_DIR = PROJECT_ROOT / "data"
CONTAINER_DATA_DIR = "/app/data"


# Text profiles every candidate is scored against. Short-first on purpose: that
# is the regime where a fixed prompt-prefill cost hurts most.
TEXT_PROFILES: dict[str, str] = {
    "short": "Buongiorno. Oggi parlo con te per un momento di silenzio e di pace.",
    "medium": (
        "Questa e una prova di sintesi vocale pensata per misurare con precisione "
        "la velocita di generazione, l'uso della memoria video e la stabilita del "
        "modello. Il sole tramonta lentamente dietro le colline mentre una brezza "
        "leggera attraversa il prato: che cosa succedera domani?"
    ),
    "long": (
        "Il sole sta lentamente tramontando dietro le colline, dipingendo il cielo "
        "di un arancio che sfuma nel viola. La campana del paese suona lontana, e "
        "attraverso i campi l'aria porta il profumo del fieno appena mietuto. Mia "
        "nonna racconta sempre che le sere d'estate duravano il doppio, quando non "
        "c'era la televisione e si aspettava che le stelle cadessero una dopo "
        "l'altra. Mi fermo un attimo, guardo il cielo che cambia, e penso che "
        "forse anche noi stiamo imparando a guardare le cose con un poco piu di "
        "calma."
    ),
}

# --------------------------------------------------------------------------- #
#  Search space
# --------------------------------------------------------------------------- #
# ``restart`` dimensions need a container restart + torch.compile warmup
# (~3-5 min each). ``restart`` dimensions are swept first because they define
# the feasible envelope; cheap dimensions are tuned inside that envelope.
#
# NOTE on max_generate_length: under OPTIMIZE=true the patch count is snapped UP
# to a compile bucket ((64, 128, 256, 512)), so only bucket-aligned values are
# offered. A value of 96 would behave exactly like 128 but validate a tighter
# prompt limit, so sweeping it would waste a restart to learn nothing.
SERVER_DIMENSIONS: list[dict[str, Any]] = [
    {
        "key": "model_name",
        "env": "TTS_MODEL_NAME",
        "values": ["dots-studio/dots.tts-mf", "dots-studio/dots.tts-soar"],
        "quick_values": ["dots-studio/dots.tts-mf"],
        # Swapped first: the two checkpoints differ by more than speed. mf is a
        # distilled 2-4 step model; soar is a continuous autoregressive model
        # that targets higher speaker similarity (published SIM > 83.9) but has
        # a much larger VRAM floor. Which one is feasible decides the feasible
        # envelope for every other dimension, so it is measured before them.
        "desc": "Checkpoint. mf = distilled/fast; soar = max similarity.",
        "variant": True,  # groups the soar-vs-mf comparison table
    },
    {
        "key": "precision",
        "env": "TTS_PRECISION",
        "values": ["bfloat16", "float16"],
        "quick_values": ["bfloat16"],
        "desc": "Weight dtype. bfloat16 preferred; float16 for older cards.",
    },
    {
        "key": "max_generate_length",
        "env": "TTS_MAX_GENERATE_LENGTH",
        "values": [64, 128, 256],
        "quick_values": [64, 128],
        "desc": "Total (prompt + generated) audio patches. 1 patch = 160 ms.",
    },
    {
        "key": "max_sequence_length",
        "env": "TTS_MAX_SEQUENCE_LENGTH",
        "values": [512, 1024, 2048],
        "quick_values": [512, 2048],
        "desc": "LLM KV-cache length in tokens.",
    },
    {
        "key": "num_steps",
        "env": "TTS_NUM_STEPS",
        "values": [1, 2, 4],
        "quick_values": [1, 2],
        "desc": "Sampling steps; also sizes the DiT KV cache.",
    },
    {
        "key": "vocoder_chunk_patches",
        "env": "TTS_VOCODER_CHUNK_PATCHES",
        "values": [8, 12, 20, 32],
        "quick_values": [12, 32],
        "desc": "Latent patches per BigVGAN call (bounds decoder activations).",
    },
    {
        "key": "optimize",
        "env": "TTS_OPTIMIZE",
        "values": ["true", "false"],
        "quick_values": ["true", "false"],
        "desc": "torch.compile + DiT KV cache. False is eager and much slower.",
    },
    {
        "key": "low_vram_patches",
        "env": "TTS_LOW_VRAM_PATCHES",
        "values": ["true", "false"],
        "quick_values": ["true"],
        "desc": "app/patches.py memory patches. Required on a 6 GB card.",
    },
]

# Free to change per request: no restart, so these are swept after the envelope
# is established and cost seconds rather than minutes.
REQUEST_DIMENSIONS: list[dict[str, Any]] = [
    {
        "key": "guidance_scale",
        "env": "TTS_GUIDANCE_SCALE",
        "values": [1.0, 1.2, 1.5],
        "quick_values": [1.2],
        "desc": "CFG strength. Affects quality, not memory.",
    },
]


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    """One measured (config, text profile) pair."""

    profile: str
    ok: bool = False
    error: str = ""
    rtf: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    audio_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "ok": self.ok,
            "error": self.error,
            "rtf": self.rtf,
            "peak_vram_mb": self.peak_vram_mb,
            "audio_seconds": self.audio_seconds,
            "generation_seconds": self.generation_seconds,
        }


@dataclass
class Candidate:
    """A fully evaluated configuration."""

    config: dict[str, str]
    label: str
    trials: list[Trial] = field(default_factory=list)
    feasible: bool = False
    rejected: str = ""
    worst_rtf: Optional[float] = None
    mean_rtf: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    usable_audio_seconds: Optional[float] = None
    score: Optional[float] = None
    skipped: bool = False

    def evaluate(self) -> None:
        """Derive the aggregate figures used for ranking.

        Must run before :meth:`Evaluator._score`, which consumes the derived
        fields and the ``feasible`` flag set here.
        """
        good = [t for t in self.trials if t.ok and t.rtf is not None]
        peaks = [
            float(t.peak_vram_mb) for t in self.trials if t.peak_vram_mb is not None
        ]
        self.peak_vram_mb = max(peaks) if peaks else None
        if not good:
            self.feasible = False
            if not self.rejected:
                first_bad = next((t for t in self.trials if not t.ok), None)
                self.rejected = (
                    (first_bad.error if first_bad else "no successful profile")
                    or "unknown failure"
                )[:300]
            return
        self.feasible = True
        rtfs = [float(t.rtf) for t in good]  # type: ignore[arg-type]
        # Worst-case ranking: a config that is fast on long text but slow on
        # short text is not a win for real use.
        self.worst_rtf = max(rtfs)
        self.mean_rtf = statistics.fmean(rtfs)
        # Longest profile that actually succeeded = the real generation ceiling.
        audios = [
            float(t.audio_seconds)
            for t in good
            if t.audio_seconds is not None
        ]
        self.usable_audio_seconds = max(audios) if audios else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "config": self.config,
            "feasible": self.feasible,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "worst_rtf": round(self.worst_rtf, 4) if self.worst_rtf else None,
            "mean_rtf": round(self.mean_rtf, 4) if self.mean_rtf else None,
            "peak_vram_mb": self.peak_vram_mb,
            "peak_vram_gb": (
                round(self.peak_vram_mb / 1024.0, 3) if self.peak_vram_mb else None
            ),
            "usable_audio_seconds": self.usable_audio_seconds,
            "score": round(self.score, 5) if self.score is not None else None,
            "trials": [t.as_dict() for t in self.trials],
        }


def repo_relative(path: "os.PathLike[str] | str") -> str:
    """Render a path for something that gets committed.

    A tracked report is read on another machine, where an absolute path is worse
    than useless: it names a home directory and a checkout layout that have
    nothing to do with the reader. ``tuning_report.json`` shipped
    ``/home/r3msi/Documents/projects/products/TTS/...`` - the operator's username,
    and the project directory name from *before* the rename.

    So: repo-relative when the file lives inside the project, absolute otherwise.
    A clip outside the tree genuinely needs its full path to mean anything, which
    is the one case where saying where it is beats hiding it.

    Deliberately local to this module rather than in ``app.config``: the tuner
    imports nothing from ``app`` today, and doing so would make a malformed
    ``.env`` raise during *its* import - turning the tool you would reach for to
    diagnose a configuration problem into a casualty of it.
    """
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_relative_to(PROJECT_ROOT):
        return str(candidate.relative_to(PROJECT_ROOT))
    return str(candidate)


@dataclass
class Reference:
    """A reference clip plus the transcript that must match it exactly."""

    name: str
    label: str
    audio: Path
    text: Path
    seconds: float = 0.0

    @property
    def patches(self) -> int:
        return int(math.ceil(self.seconds / 0.16)) if self.seconds else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            # Repo-relative on purpose: this dict lands in tuning_report.json and
            # best_config.json, which are committed, and an absolute path there
            # bakes the operator's home directory and checkout layout into a
            # public repository.
            "audio": repo_relative(self.audio),
            "seconds": round(self.seconds, 2),
            "patches": self.patches,
        }


def load_reference_set(path: Path) -> list[Reference]:
    """Read a reference-set JSON file (see data/input/reference_set.json)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("references") if isinstance(payload, dict) else payload
    if not entries:
        raise ValueError(f"{path} contains no 'references' entries")
    references: list[Reference] = []
    for entry in entries:
        audio = (PROJECT_ROOT / str(entry["audio"])).resolve()
        text = (PROJECT_ROOT / str(entry["text"])).resolve()
        if not audio.exists():
            raise FileNotFoundError(f"reference audio not found: {audio}")
        if not text.exists():
            raise FileNotFoundError(f"reference transcript not found: {text}")
        references.append(
            Reference(
                name=str(entry.get("name") or audio.stem),
                label=str(entry.get("label") or audio.stem),
                audio=audio,
                text=text,
                seconds=load_reference_seconds(audio),
            )
        )
    return references


# --------------------------------------------------------------------------- #
#  Environment / compose plumbing
# --------------------------------------------------------------------------- #
def parse_env_file(path: Path) -> dict[str, str]:
    """Read a ``.env`` file into a dict (comments and blanks ignored)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def compose_command() -> list[str]:
    return ["docker", "compose"]


def run_compose(
    args: list[str],
    env_overrides: dict[str, str],
    *,
    timeout: float = 900.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``docker compose`` with extra environment variables.

    Compose interpolation reads the shell environment first and falls back to
    ``.env``, so exporting a variable here overrides the file **without**
    modifying it. That is what lets the tuner vary settings per candidate.
    """
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    result = subprocess.run(
        compose_command() + args,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError(
            f"docker compose {' '.join(args)} failed "
            f"({result.returncode}):\n" + "\n".join(tail)
        )
    return result


def http_json(url: str, timeout: float) -> dict[str, Any]:
    """GET a JSON document, raising on transport or decoding failure."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def wait_for_ready(
    api_url: str,
    timeout: float,
    poll: float = 5.0,
    *,
    on_poll: Optional[Any] = None,
) -> dict[str, Any]:
    """Block until ``/health`` reports ``ready``.

    A checkpoint that OOMs during warmup never becomes ready: the API keeps
    answering 503 with the load error in ``model.error``. We surface that as a
    normal candidate failure rather than a crash, so the sweep can move on.
    """
    deadline = time.monotonic() + timeout
    url = f"{api_url.rstrip('/')}/health"
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = http_json(url, timeout=min(30.0, max(5.0, timeout / 6)))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            time.sleep(poll)
            continue
        if last.get("ready"):
            return last
        error = ((last.get("model") or {}).get("error")) or ""
        if error:
            # A hard load failure will not resolve itself; stop early.
            raise RuntimeError(f"checkpoint load failed: {error}")
        if on_poll is not None:
            on_poll(last)
        time.sleep(poll)
    raise TimeoutError(f"engine not ready after {timeout:.0f}s")


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #
class Evaluator:
    """Applies a configuration, measures it, and keeps the resulting records."""

    def __init__(
        self, args: argparse.Namespace, output_dir: Path, references: list[Reference]
    ) -> None:
        self.args = args
        self.output_dir = output_dir
        self.api_url = args.api_url.rstrip("/")
        self.records: list[Candidate] = []
        self.references = references
        # An empty set is tolerated so the pure scoring / cost-model logic can be
        # exercised without any audio on disk.
        self.active = references[0] if references else None
        self._current: Optional[dict[str, str]] = None
        self._scratch = 0
        # Refreshed from /health on every restart; 0.16 s is the dots.tts default.
        self.patch_seconds = 0.16

    # -- lifecycle ---------------------------------------------------------- #
    def apply_config(self, config: dict[str, str]) -> None:
        """Recreate the container with ``config`` and wait until it is ready."""
        if self._current == config:
            return  # already running with these settings
        run_compose(["up", "-d", "--force-recreate"], config)
        try:
            health = wait_for_ready(self.api_url, self.args.ready_timeout)
        finally:
            self._announce_ready()
        model = health.get("model") or {}
        patch_seconds = model.get("audio_patch_seconds")
        if patch_seconds:
            self.patch_seconds = float(patch_seconds)
        self._current = dict(config)
        self._verify_effective(config, model)

    def _announce_ready(self) -> None:
        if not self.args.quiet:
            print("    engine ready", flush=True)

    @staticmethod
    def _container_path(host_path: Path) -> str:
        """Translate a host path under ``data/`` to its in-container location.

        Falls back to the absolute host path when the file lives outside the
        bind mount, which keeps the failure obvious rather than silent.
        """
        try:
            relative = host_path.resolve().relative_to(DATA_DIR.resolve())
        except ValueError:
            return str(host_path)
        return f"{CONTAINER_DATA_DIR}/{relative.as_posix()}"

    def _verify_effective(
        self, requested: dict[str, str], model: dict[str, Any]
    ) -> None:
        """Warn when the server did not apply what we asked for.

        ``/health`` is the ground truth -- it reports ``settings.model`` from
        inside the container -- so a clamped or silently ignored setting shows up
        here instead of quietly skewing the comparison.
        """
        mapping = {
            "TTS_MAX_GENERATE_LENGTH": model.get("max_generate_length"),
            "TTS_NUM_STEPS": model.get("default_num_steps"),
            "TTS_OPTIMIZE": model.get("optimize"),
            "TTS_PRECISION": model.get("precision"),
        }
        for env_key, actual in mapping.items():
            want = requested.get(env_key)
            if want is None or actual is None:
                continue
            if str(actual).lower() != str(want).lower():
                print(
                    f"  ! {env_key}: requested {want!r} but the server reports "
                    f"{actual!r} (clamped or ignored)"
                )

    # -- measurement -------------------------------------------------------- #
    def measure(self, config: dict[str, str], label: str) -> Candidate:
        """Score one configuration across every text profile."""
        record = Candidate(config=dict(config), label=label)
        self.records.append(record)

        try:
            self.apply_config(config)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
            record.rejected = f"restart/load failed: {exc}"[:300]
            print(f"  [skip] {label}: {record.rejected}")
            return record

        for profile in self.args.profiles:
            trial = self._run_profile(profile)
            record.trials.append(trial)
            if trial.ok:
                print(
                    f"    {profile:<6} rtf {trial.rtf:<7} peak "
                    f"{(trial.peak_vram_mb or 0) / 1024:.2f} GB  audio "
                    f"{trial.audio_seconds}s"
                )
            else:
                print(f"    {profile:<6} FAILED: {trial.error[:120]}")
                # A resource failure on one profile disqualifies the candidate:
                # the next, longer profile could only fail harder. Measuring it
                # would just burn minutes for a foregone conclusion.
                if self._is_resource_error(trial.error):
                    break

        record.evaluate()
        self._score(record)
        self._print_verdict(record)
        self._dump_progress()
        return record

    def _is_resource_error(self, error: str) -> bool:
        lowered = error.lower()
        return any(
            token in lowered
            for token in ("outofmemory", "out of memory", "cuda", "memoryerror")
        )

    def _run_profile(self, profile: str) -> Trial:
        """Run the acceptance benchmark for one text length and harvest it.

        The benchmark executes *inside* the container, so it is handed container
        paths; the report is then read back from the host side of the bind
        mount. Passing the host path to the container silently writes a file
        that does not exist where we look for it.
        """
        trial = Trial(profile=profile)
        self._scratch += 1
        name = f"tune_{self._scratch:04d}"
        report_path = self.output_dir / f"{name}.json"
        wav_path = self.output_dir / f"{name}.wav"
        container_report = self._container_path(report_path)
        container_wav = self._container_path(wav_path)
        command = [
            "exec", "-T", "tts", "python", "scripts/benchmark_test.py",
            "--api-url", self.api_url,
            # The benchmark opens the reference itself, inside the container, so
            # these must be container paths for the same reason as --report.
            "--reference-audio", self._container_path(self.active.audio),
            "--reference-text", self._container_path(self.active.text),
            "--text", TEXT_PROFILES[profile],
            "--samples", str(self.args.samples),
            "--warmup", str(self.args.warmup),
            "--report", container_report,
            "--output", container_wav,
            # An infeasible candidate is data, not a hard stop.
            "--no-fail",
        ]
        try:
            result = run_compose(
                command, self._current or {}, timeout=self.args.request_timeout * 2
            )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            trial.error = str(exc)
            return trial
        if result.returncode != 0:
            trial.error = (result.stderr or result.stdout or "").strip()[-400:]
            return trial
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            trial.error = f"could not read benchmark report: {exc}"
            return trial

        results = report.get("results") or {}
        samples = report.get("samples") or []
        if not results.get("http_success"):
            failure = next(
                (s.get("error") for s in samples if s.get("http_status") != 200), ""
            )
            trial.error = (failure or "no successful HTTP call")[:400]
            return trial
        trial.ok = True
        trial.rtf = results.get("rtf_mean")
        trial.peak_vram_mb = results.get("peak_vram_mb")
        trial.audio_seconds = results.get("audio_seconds_mean")
        trial.generation_seconds = results.get("generation_seconds_mean")
        return trial

    # -- scoring ------------------------------------------------------------ #
    def prompt_patches(self, reference_seconds: float) -> int:
        """Reference prompt cost in audio patches (mirrors process_txt.py)."""
        return int(math.ceil(reference_seconds / self.patch_seconds))

    def preflight(self, config: dict[str, str], reference: Reference) -> str:
        """Reject a config that cannot possibly serve the reference clip.

        ``_prepare_inputs`` in the runtime raises when the prompt patch count
        reaches ``max_generate_length``, and the whole request is wasted
        otherwise. Catching it here avoids paying a ~4 min restart to be told so.
        """
        cap = config.get("TTS_MAX_GENERATE_LENGTH")
        if not cap or not self.patch_seconds:
            return ""
        try:
            cap_value = int(cap)
        except ValueError:
            return ""
        prompt = self.prompt_patches(reference.seconds)
        if prompt >= cap_value:
            return (
                f"{reference.name} needs {prompt} patches but the cap is "
                f"{cap_value} (no room to generate)"
            )
        return ""

    def _score(self, record: Candidate) -> None:
        """Rank feasible candidates; lower is better.

        Hard gates come first (VRAM ceiling, RTF target) because they are the
        spec's acceptance criteria. The margin is deliberately treated as a
        *gate*, not a bonus, so a config that only just fits is not selected when
        a safer alternative exists.
        """
        if not record.feasible or record.peak_vram_mb is None or record.worst_rtf is None:
            record.score = None
            return

        limit_mb = (self.args.vram_limit_gb - self.args.vram_margin_gb) * 1024.0
        if record.peak_vram_mb > limit_mb:
            record.feasible = False
            record.rejected = (
                f"peak {record.peak_vram_mb / 1024:.2f} GB exceeds the "
                f"{self.args.vram_limit_gb - self.args.vram_margin_gb:.2f} GB budget "
                f"(limit {self.args.vram_limit_gb} GB - margin "
                f"{self.args.vram_margin_gb} GB)"
            )
            record.score = None
            return
        if record.worst_rtf > self.args.rtf_target:
            record.feasible = False
            record.rejected = (
                f"worst-profile RTF {record.worst_rtf:.3f} exceeds the target "
                f"{self.args.rtf_target}"
            )
            record.score = None
            return

        rtf_term = record.worst_rtf
        headroom_gb = (self.args.vram_limit_gb - self.args.vram_margin_gb) - (
            record.peak_vram_mb / 1024.0
        )
        if self.args.objective == "throughput":
            # Pure speed, with a light nudge away from the VRAM wall.
            record.score = rtf_term + 0.02 * (1.0 - max(0.0, headroom_gb))
        elif self.args.objective == "headroom":
            # Longest audio per request wins; speed breaks ties.
            record.score = -(record.usable_audio_seconds or 0.0) + rtf_term
        else:  # balanced
            # Speed first, but reward usable audio length and VRAM headroom.
            record.score = (
                rtf_term
                - 0.004 * min(60.0, record.usable_audio_seconds or 0.0)
                - 0.01 * min(1.0, headroom_gb)
            )

    def _print_verdict(self, record: Candidate) -> None:
        if record.feasible:
            print(
                f"  => FEASIBLE  worst-rtf {record.worst_rtf:.3f}  "
                f"peak {record.peak_vram_mb / 1024:.2f} GB  "
                f"audio {record.usable_audio_seconds}s  score {record.score:.4f}"
            )
        else:
            print(f"  => REJECTED  {record.rejected}")

    def _dump_progress(self) -> None:
        """Persist after every candidate so ``--resume`` survives a crash."""
        path = self.output_dir / "tuning_report.json"
        try:
            path.write_text(
                json.dumps(
                    {
                        "updated": datetime.now(timezone.utc).isoformat(),
                        "candidates": [r.as_dict() for r in self.records],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - disk issues
            print(f"  (could not write progress: {exc})", file=sys.stderr)


# --------------------------------------------------------------------------- #
#  Sweep driver
# --------------------------------------------------------------------------- #
def load_reference_seconds(path: Path) -> float:
    """Duration of the reference WAV, via the stdlib ``wave`` module."""
    import wave

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or 1
        return handle.getnframes() / float(rate)


def load_resume(results_path: Path) -> dict[str, dict[str, Any]]:
    """Map ``label -> finished candidate`` from a previous run."""
    if not results_path.exists():
        return {}
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    done: dict[str, dict[str, Any]] = {}
    for record in payload.get("candidates") or []:
        if record.get("label"):
            done[str(record["label"])] = record
    return done


def candidate_label(config: dict[str, str], dimension: str) -> str:
    return f"{dimension}={config.get(dimension_env(dimension), '?')}"


def dimension_env(dimension: str) -> str:
    for group in (SERVER_DIMENSIONS, REQUEST_DIMENSIONS):
        for entry in group:
            if entry["key"] == dimension:
                return str(entry["env"])
    return dimension


def restore_candidate(
    config: dict[str, str], label: str, payload: dict[str, Any]
) -> Candidate:
    """Rebuild a :class:`Candidate` from a previously written report entry."""
    record = Candidate(config=dict(config), label=label)
    for raw in payload.get("trials") or []:
        trial = Trial(
            profile=str(raw.get("profile", "?")),
            ok=bool(raw.get("ok")),
            error=str(raw.get("error") or ""),
            rtf=raw.get("rtf"),
            peak_vram_mb=raw.get("peak_vram_mb"),
            audio_seconds=raw.get("audio_seconds"),
            generation_seconds=raw.get("generation_seconds"),
        )
        record.trials.append(trial)
    record.feasible = bool(payload.get("feasible"))
    record.rejected = str(payload.get("rejected") or "")
    record.worst_rtf = payload.get("worst_rtf")
    record.mean_rtf = payload.get("mean_rtf")
    record.peak_vram_mb = payload.get("peak_vram_mb")
    record.usable_audio_seconds = payload.get("usable_audio_seconds")
    record.score = payload.get("score")
    return record



def sweep_reference(
    evaluator: Evaluator,
    base: dict[str, str],
    *,
    quick: bool,
    resume: dict[str, dict[str, Any]],
) -> Reference:
    """Choose the reference clip, which is usually the dominant RTF lever.

    The prompt prefill is paid on every request regardless of output length, so
    for short utterances its cost dominates. No env setting shrinks it; only a
    shorter reference does. Swept first so the remaining dimensions are tuned
    against the reference we will actually ship.
    """
    references = evaluator.references
    if len(references) == 1:
        print("  (single reference configured; nothing to sweep)\n")
        return references[0]

    print("=" * 74)
    print("  dimension: reference clip   (fixed per-request prefill cost)")
    print("=" * 74)

    best: Optional[Reference] = None
    best_score: Optional[float] = None
    for reference in references:
        evaluator.active = reference
        reason = evaluator.preflight(base, reference)
        if reason:
            record = Candidate(config=dict(base), label=f"reference={reference.name}")
            record.rejected = f"pre-flight: {reason}"
            record.skipped = True
            evaluator.records.append(record)
            print(f"  [skip] {reference.name:<8} {reason}")
            continue

        label = f"reference={reference.name}"
        cached = resume.get(label)
        if cached is not None and cached.get("trials"):
            record = restore_candidate(dict(base), label, cached)
            evaluator.records.append(record)
            print(f"  [cache] {reference.name:<8} reused from a previous run")
        else:
            record = evaluator.measure(dict(base), label)

        print(
            f"  {reference.name:<8} {reference.seconds:5.2f}s "
            f"({reference.patches:>3} patches)"
        )
        if record.feasible and record.score is not None:
            if best_score is None or record.score < best_score:
                best_score, best = record.score, reference

    if best is None:
        print("  !! no reference met the budget; keeping the current one\n")
        return references[0]
    evaluator.active = best
    print(f"  -> keeping {best.name} ({best.label}, {best.patches} patches)\n")
    return best


def sweep_dimension(
    evaluator: Evaluator,
    base: dict[str, str],
    dimension: dict[str, Any],
    *,
    quick: bool,
    resume: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Try every value of one dimension, keep the best, return the new base."""
    env_key = str(dimension["env"])
    values = list(
        dimension["quick_values"] if quick else dimension["values"]
    )
    current = base.get(env_key)
    if current is not None and str(current) not in [str(v) for v in values]:
        values.insert(0, current)  # always keep the incumbent measurable

    print("=" * 74)
    print(f"  dimension: {env_key}   {dimension['desc']}")
    print("=" * 74)

    best_config = dict(base)
    best_score: Optional[float] = None
    best_label: Optional[str] = None

    for value in values:
        trial_config = dict(base)
        trial_config[env_key] = str(value)
        label = f"{dimension['key']}={value}"

        reason = evaluator.preflight(trial_config, evaluator.active)
        if reason:
            record = Candidate(config=trial_config, label=label)
            record.rejected = f"pre-flight: {reason}"
            record.skipped = True
            evaluator.records.append(record)
            print(f"  [skip] {label}: {reason}")
            continue

        cached = resume.get(label)
        if cached is not None and cached.get("trials"):
            record = restore_candidate(trial_config, label, cached)
            evaluator.records.append(record)
            print(f"  [cache] {label}: reused from a previous run")
        else:
            record = evaluator.measure(trial_config, label)

        if not record.feasible or record.score is None:
            continue
        if best_score is None or record.score < best_score:
            best_score = record.score
            best_config = trial_config
            best_label = label

    if best_label is None:
        print(f"  !! no feasible value for {env_key}; keeping the incumbent")
        return base
    print(f"  -> keeping {best_label} (score {best_score:.4f})\n")
    return best_config


def fit_cost_model(
    evaluator: "Evaluator",
) -> Optional[dict[str, float]]:
    """Least-squares fit of ``generation = fixed + rate * audio_seconds``.

    Measured across every successful trial, this separates the per-request
    fixed cost from the marginal per-second cost. It is what turns "no
    candidate passed" into an actionable answer: when ``fixed`` dominates, the
    only lever is longer audio per request, and no setting will help.
    """
    points: list[tuple[float, float]] = []
    for record in evaluator.records:
        for trial in record.trials:
            if (
                trial.ok
                and trial.audio_seconds
                and trial.generation_seconds
                and trial.audio_seconds > 0
            ):
                points.append((float(trial.audio_seconds), float(trial.generation_seconds)))
    if len(points) < 2:
        return None
    n = float(len(points))
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-9:
        return None
    rate = (n * sum_xy - sum_x * sum_y) / denominator
    fixed = (sum_y - rate * sum_x) / n
    if rate <= 0:  # non-monotonic data; the model is not meaningful
        return None
    return {"fixed_seconds": fixed, "rate_seconds_per_audio": rate, "points": n}


def minimum_audio_for_rtf(model: dict[str, float], target: float) -> Optional[float]:
    """Audio length at which ``RTF <= target`` under the fitted model.

    ``(fixed + rate * A) / A <= target``  =>  ``A >= fixed / (target - rate)``.
    Returns ``None`` when the asymptotic rate already meets the target.
    """
    rate = model["rate_seconds_per_audio"]
    if rate >= target:
        return None
    return model["fixed_seconds"] / (target - rate)


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def write_report(
    path: Path,
    evaluator: Evaluator,
    args: argparse.Namespace,
    winner: Optional[dict[str, str]],
    reference: Reference,
) -> None:
    """Write the full matrix, including every rejected candidate."""
    feasible = [r for r in evaluator.records if r.feasible and r.score is not None]
    feasible.sort(key=lambda r: float(r.score or 0.0))
    model = fit_cost_model(evaluator)
    minimum_audio = minimum_audio_for_rtf(model, args.rtf_target) if model else None
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "objective": args.objective,
        "profiles": list(args.profiles),
        "cost_model": model,
        "minimum_audio_seconds_for_rtf_target": minimum_audio,
        "thresholds": {
            "vram_limit_gb": args.vram_limit_gb,
            "vram_margin_gb": args.vram_margin_gb,
            "rtf_target": args.rtf_target,
        },
        "reference": {
            **reference.as_dict(),
            "patch_seconds": evaluator.patch_seconds,
        },
        "reference_set": [r.as_dict() for r in evaluator.references],
        "quick_mode": args.quick,
        "winner": winner,
        "ranking": [r.as_dict() for r in feasible],
        "candidates": [r.as_dict() for r in evaluator.records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tuned_env(
    path: Path,
    winner: dict[str, str],
    evaluator: Evaluator,
    args: argparse.Namespace,
    reference: Reference,
) -> None:
    """Emit the winning settings as a ready-to-paste ``.env`` block."""
    tuned = [
        d for d in SERVER_DIMENSIONS + REQUEST_DIMENSIONS
        if d["env"] in winner
    ]
    lines = [
        "# Generated by scripts/autotune.py",
        f"# {datetime.now(timezone.utc).isoformat()}",
        f"# objective={args.objective}  profiles={','.join(args.profiles)}",
        f"# budget: peak VRAM < {args.vram_limit_gb} GB "
        f"(margin {args.vram_margin_gb} GB), worst-profile RTF <= {args.rtf_target}",
        f"# reference: {reference.name} ({reference.label}) {reference.seconds:.2f}s "
        f"= {reference.patches} patches -> {reference.audio.name}",
        f"# use it with: --reference-audio {reference.audio} "
        f"--reference-text {reference.text}",
        "",
    ]
    for entry in tuned:
        lines.append(f"# {entry['desc']}")
        lines.append(f"{entry['env']}={winner[entry['env']]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_best_config(
    path: Path,
    winner: dict[str, str],
    evaluator: "Evaluator",
    args: argparse.Namespace,
    reference: Reference,
) -> None:
    """Emit the winning profile as ``best_config.json`` (v3 Task 4.3).

    This is the file ``app/config.py`` reads at startup, so it carries only the
    tuning dimensions - never measured values. Numbers are cast from the
    ``TTS_NUM_STEPS=2`` strings the sweep works in to real ints/floats, and flags
    to real booleans, so the file is self-describing.
    """
    keys = [str(d["env"]) for d in SERVER_DIMENSIONS + REQUEST_DIMENSIONS]
    int_keys = {"TTS_NUM_STEPS", "TTS_MAX_GENERATE_LENGTH", "TTS_MAX_SEQUENCE_LENGTH",
                "TTS_VOCODER_MERGE_STEPS", "TTS_VOCODER_CHUNK_PATCHES", "TTS_SEED"}
    float_keys = {"TTS_GUIDANCE_SCALE", "TTS_SPEAKER_SCALE", "TTS_VRAM_LIMIT_GB",
                  "TTS_RTF_TARGET"}
    bool_keys = {"TTS_OPTIMIZE", "TTS_LOW_VRAM_PATCHES", "TTS_WARMUP_ON_OPTIMIZE"}

    settings: dict[str, Any] = {}
    for key, raw in winner.items():
        if key not in keys:
            continue
        value: Any = raw
        try:
            if key in int_keys:
                value = int(float(raw))
            elif key in float_keys:
                value = float(raw)
            elif key in bool_keys:
                value = str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        except (TypeError, ValueError):
            logger_value = raw
            value = logger_value  # keep the string; the service will fall back
        settings[key] = value

    # A config can be measured more than once in one sweep: the incumbent
    # baseline is evaluated (and often rejected) before the dimension that
    # already holds it is reached, at which point the same config is evaluated
    # again and wins. Picking the *first* match would therefore attach the
    # baseline's rejected numbers - a null score and a worst_rtf above the
    # target - to a profile that actually passed. Prefer the feasible record.
    matching = [r for r in evaluator.records if r.config == winner]
    record = next((r for r in matching if r.feasible), None) or (
        matching[0] if matching else None
    )
    payload: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/autotune.py",
        "host": _host_description(),
        "objective": args.objective,
        "constraints": {
            "vram_limit_gb": args.vram_limit_gb,
            "vram_margin_gb": args.vram_margin_gb,
            "rtf_target": args.rtf_target,
        },
        "reference": {
            "name": reference.name,
            "label": reference.label,
            "seconds": round(reference.seconds, 3),
            "patches": reference.patches,
        },
        "settings": settings,
    }
    if record is not None:
        payload["measured"] = {
            "score": record.score,
            "peak_vram_mb": record.peak_vram_mb,
            "worst_rtf": record.worst_rtf,
            "usable_audio_seconds": record.usable_audio_seconds,
        }
    payload["variants"] = _variant_comparison(evaluator, winner)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _host_description() -> dict[str, Any]:
    """Best-effort identification of the machine the profile was measured on."""
    info: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = round(props.total_memory / (1024 * 1024), 1)
            info["torch"] = torch.__version__
    except Exception:  # pragma: no cover - the client may not have torch
        info["gpu"] = None
    return info

def _variant_comparison(
    evaluator: "Evaluator", winner: dict[str, str]
) -> list[dict[str, Any]]:
    """soar vs mf, as measured (v3 Task 4.3's comparison table).

    Keeps the checkpoints that failed, not just the ones that worked: "soar
    cannot run on this GPU" is the single most useful thing this sweep can
    report, and it is only visible if the losers stay in the table.
    """
    rows: dict[str, dict[str, Any]] = {}
    for record in evaluator.records:
        name = record.config.get("TTS_MODEL_NAME")
        if not name:
            continue
        row = rows.setdefault(
            name,
            {
                "model": name,
                "attempts": 0,
                "feasible": 0,
                "best_rtf": None,
                "min_peak_vram_mb": None,
                "best_score": None,
                "is_winner": False,  # set below; the first record is often the
                # rejected incumbent baseline, not the winning measurement
                "last_error": None,
            },
        )
        row["attempts"] += 1
        if record.config == winner:
            # Marked here rather than at setdefault(): the winner config is
            # usually measured twice, and only the later, feasible pass counts.
            row["is_winner"] = True
        if not record.feasible:
            row["last_error"] = (record.rejected or row["last_error"] or "")[:200] or None
            continue
        row["feasible"] += 1
        if record.worst_rtf is not None:
            row["best_rtf"] = (
                record.worst_rtf
                if row["best_rtf"] is None
                else min(row["best_rtf"], record.worst_rtf)
            )
        if record.peak_vram_mb is not None:
            row["min_peak_vram_mb"] = (
                record.peak_vram_mb
                if row["min_peak_vram_mb"] is None
                else min(row["min_peak_vram_mb"], record.peak_vram_mb)
            )
        if record.score is not None:
            row["best_score"] = (
                record.score
                if row["best_score"] is None
                else min(row["best_score"], record.score)
            )
    return sorted(rows.values(), key=lambda r: r["model"])


def print_variant_table(evaluator: "Evaluator", args: argparse.Namespace) -> None:
    """Human-readable soar-vs-mf summary printed at the end of a sweep."""
    rows = _variant_comparison(evaluator, evaluator.active or {})
    if not rows:
        return
    print("\n[tune] model comparison (v3 Task 4.3)")
    print(f"  {'checkpoint':<32} {'feasible':>11} {'best RTF':>9} {'min peak':>11}")
    print(f"  {'-' * 32} {'-' * 11} {'-' * 9} {'-' * 11}")
    for row in rows:
        rtf = f"{row['best_rtf']:.4f}" if row["best_rtf"] is not None else "-"
        peak = (
            f"{row['min_peak_vram_mb']:.0f} MB"
            if row["min_peak_vram_mb"] is not None
            else "-"
        )
        mark = "   <- selected" if row.get("is_winner") else ""
        print(
            f"  {row['model']:<32} {row['feasible']}/{row['attempts']:<9} "
            f"{rtf:>9} {peak:>11}{mark}"
        )
    for row in rows:
        if row["feasible"] == 0:
            budget = f"the {args.vram_limit_gb} GB peak-VRAM budget"
            print(
                f"  ruled out: {row['model']} never fit in {budget}"
                + (f" - {row['last_error']}" if row.get("last_error") else "")
            )


def apply_to_env(
    winner: dict[str, str], env_path: Path, keys: list[str]
) -> Path:
    """Rewrite the tuned keys in ``.env``, keeping a timestamped backup.

    Existing comments and unrelated settings are preserved; only the tuned
    ``KEY=value`` lines are replaced (or appended when absent).
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = env_path.with_suffix(f".env.bak-{stamp}")
    shutil.copy2(env_path, backup)

    original = env_path.read_text(encoding="utf-8").splitlines()
    remaining = dict(winner)
    output: list[str] = []
    for line in original:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)
    if remaining:
        output.append("")
        output.append("# --- added by scripts/autotune.py ---")
        for key, value in remaining.items():
            output.append(f"{key}={value}")
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return backup


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep dots.tts runtime settings, measure VRAM/RTF, and emit the "
            "best configuration for this GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Each server-side candidate costs a container restart plus a "
            "torch.compile warmup (~3-5 min). Budget accordingly."
        ),
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--reference-audio", default="data/input/reference_clean.wav"
    )
    parser.add_argument(
        "--reference-set",
        default="data/input/reference_set.json",
        help=(
            "JSON file listing reference clips to choose between (see "
            "data/input/reference_set.json). The reference length is a fixed "
            "per-request prefill cost and is usually the dominant RTF lever, so "
            "it is swept first. Pass a file with one entry to disable."
        ),
    )
    parser.add_argument(
        "--no-sweep-reference",
        action="store_true",
        help="Keep the single --reference-audio clip instead of sweeping the set.",
    )
    parser.add_argument(
        "--objective",
        choices=("balanced", "throughput", "headroom"),
        default="balanced",
        help=(
            "balanced: speed with VRAM headroom (default). "
            "throughput: lowest RTF. "
            "headroom: longest usable audio per request."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["short", "medium", "long"],
        choices=sorted(TEXT_PROFILES),
        help="Text lengths to score each candidate against.",
    )
    parser.add_argument("--vram-limit-gb", type=float, default=5.5)
    parser.add_argument(
        "--vram-margin-gb",
        type=float,
        default=0.1,
        help=(
            "Safety headroom below the VRAM limit, applied as a hard gate so a "
            "config that only just fits loses to a safer one. The default is "
            "0.1 GB because the verified 6 GB configuration peaks at ~5.36 GB: a "
            "larger margin (e.g. 0.2) would reject it outright."
        ),
    )
    parser.add_argument("--rtf-target", type=float, default=0.5)
    parser.add_argument(
        "--samples", type=int, default=2, help="Measured calls per profile."
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grids, fewer text profiles and one sample per profile.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse candidates already present in the previous report.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["reference"] + [d["key"] for d in SERVER_DIMENSIONS + REQUEST_DIMENSIONS],
        help=(
            "Restrict the sweep to these dimensions. 'reference' selects the "
            "reference clip; it is swept implicitly when --only is omitted."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the winner into .env (a timestamped backup is kept).",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Leave the container on the winning config (default: restore .env).",
    )
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _estimate_minutes(args: argparse.Namespace, dimensions: list[dict[str, Any]]) -> int:
    """Rough wall-clock estimate, so a long sweep is not a surprise.

    Counts only the dimensions actually selected, so ``--only`` narrows the
    estimate the way it narrows the work.
    """
    restarts = 0.0
    measurements = 0.0
    for dimension in dimensions:
        count = len(dimension["quick_values" if args.quick else "values"])
        if dimension in SERVER_DIMENSIONS:
            # Container restart + checkpoint load + torch.compile warmup.
            restarts += count * 4.0
            measurements += count
    per_candidate = len(args.profiles) * (args.samples + args.warmup) * 12.0 / 60.0
    return int(restarts + measurements * per_candidate + 2)


def _restore(original: dict[str, str]) -> None:
    """Put the service back on the settings it had before the sweep."""
    try:
        run_compose(["up", "-d", "--force-recreate"], original, timeout=900.0)
        wait_for_ready("http://127.0.0.1:8000", 1800.0)
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
        print(f"  (could not restore: {exc})", file=sys.stderr)


def _print_cost_model(evaluator: Evaluator, args: argparse.Namespace) -> None:
    """Explain a failed sweep with the measured fixed/marginal cost split.

    A large per-request fixed cost means short utterances cannot reach the RTF
    target at *any* setting, which is far more useful to report than "nothing
    passed" -- it tells the user to change the workload, not the config.
    """
    model = fit_cost_model(evaluator)
    if not model:
        return
    fixed = model["fixed_seconds"]
    rate = model["rate_seconds_per_audio"]
    print()
    print("  Why nothing passed - measured cost model")
    print(f"    generation = {fixed:.2f}s fixed + {rate:.3f}s per second of audio")
    print(f"    fitted on {int(model['points'])} successful call(s)")
    print(f"    => RTF = {rate:.3f} + {fixed:.2f} / audio_seconds")
    if rate >= args.rtf_target:
        print(
            f"    => the marginal rate alone ({rate:.3f}) already exceeds the "
            f"{args.rtf_target} target: unreachable on this GPU."
        )
        return
    need = minimum_audio_for_rtf(model, args.rtf_target)
    if need is not None:
        print(
            f"    => RTF <= {args.rtf_target} needs >= {need:.1f}s of audio per "
            "request."
        )
        print(
            "       The fixed cost is per request and independent of the "
            "reference length, so shorter speech cannot reach the target at any "
            "setting. Either accept a higher RTF for short text, or batch "
            "several sentences into one request."
        )


def _print_summary(
    evaluator: Evaluator,
    feasible: list[Candidate],
    best: Optional[dict[str, str]],
    report_path: Path,
    tuned_env_path: Path,
    args: argparse.Namespace,
) -> None:
    print("\n" + "=" * 74)
    print("  Summary")
    print("=" * 74)
    if not feasible:
        print("  No candidate met both budgets.")
        print(
            f"  Tried {len(evaluator.records)} configurations. Loosen "
            "--vram-margin-gb or --rtf-target to see near-misses."
        )
        _print_cost_model(evaluator, args)
        print(f"  report : {report_path}")
        print("=" * 74)
        return

    print(f"  {'rank':<5}{'config':<34}{'worst rtf':>11}{'peak GB':>10}{'audio s':>9}")
    print("  " + "-" * 69)
    for index, record in enumerate(feasible[:8], start=1):
        print(
            f"  {index:<5}{record.label:<34}"
            f"{record.worst_rtf:>11.3f}{record.peak_vram_mb / 1024:>10.2f}"
            f"{record.usable_audio_seconds:>9.1f}"
        )
    print("  " + "-" * 69)
    winner = feasible[0]
    print(f"  best   : {winner.label}")
    print(
        f"  worst-profile RTF {winner.worst_rtf:.3f} | "
        f"peak {winner.peak_vram_mb / 1024:.2f} GB | "
        f"up to {winner.usable_audio_seconds:.1f}s of audio per request"
    )
    rejected = [r for r in evaluator.records if not r.feasible]
    if rejected:
        print(f"  rejected: {len(rejected)} candidate(s) - see the report for reasons")
    print(f"  report  : {report_path}")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Progress matters over a multi-minute sweep, and stdout is block-buffered
    # whenever the output is redirected to a log file rather than a terminal.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):  # pragma: no cover - exotic stdout
        pass
    if args.quick:
        args.profiles = [p for p in args.profiles if p != "medium"] or ["short"]
        args.samples = 1

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tuning_report.json"
    tuned_env_path = output_dir / "tuned.env"
    best_config_path = output_dir / "best_config.json"

    reference_path = (PROJECT_ROOT / args.reference_audio).resolve()
    if not reference_path.exists():
        print(
            f"[tune] reference not found: {reference_path}\n"
            "       Run scripts/clean_audio.py first.",
            file=sys.stderr,
        )
        return 2

    references: list[Reference] = [
        Reference(
            name=reference_path.stem,
            label="configured clip",
            audio=reference_path,
            text=reference_path.with_suffix(".txt"),
            seconds=load_reference_seconds(reference_path),
        )
    ]
    if not args.no_sweep_reference:
        set_path = (PROJECT_ROOT / args.reference_set).resolve()
        if set_path.exists():
            try:
                references = load_reference_set(set_path)
            except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
                print(
                    f"[tune] ignoring --reference-set {set_path}: {exc}\n"
                    "       falling back to the single --reference-audio clip.",
                    file=sys.stderr,
                )
        elif args.reference_set:
            print(
                f"[tune] no reference set at {set_path}; using the single clip.",
                file=sys.stderr,
            )
    for reference in references:
        if not reference.text.exists():
            print(
                f"[tune] reference transcript not found: {reference.text}",
                file=sys.stderr,
            )
            return 2

    baseline = parse_env_file(ENV_PATH)
    if not baseline:
        print(f"[tune] could not read {ENV_PATH}", file=sys.stderr)
        return 2
    original_config = dict(baseline)

    dimensions = [
        d for d in SERVER_DIMENSIONS + REQUEST_DIMENSIONS
        if not args.only or d["key"] in args.only
    ]
    # The reference sweep runs unless it was explicitly excluded, and is skipped
    # entirely when --only names some other dimension.
    sweep_ref = (not args.no_sweep_reference) and (
        not args.only or "reference" in args.only
    )

    print("=" * 74)
    print("  dots.tts settings autotuner")
    print("=" * 74)
    print(f"  api url   : {args.api_url}")
    print(
        f"  reference : {len(references)} clip(s) - "
        + ", ".join(f"{r.name} {r.seconds:.1f}s/{r.patches}p" for r in references)
    )
    print(f"  objective : {args.objective}")
    print(f"  profiles  : {', '.join(args.profiles)}")
    print(
        f"  budget    : peak < {args.vram_limit_gb} GB "
        f"(margin {args.vram_margin_gb} GB), RTF <= {args.rtf_target}"
    )
    print(f"  mode      : {'quick' if args.quick else 'full'}")
    print(f"  dimensions: {', '.join(d['key'] for d in dimensions)}")
    print(f"  estimated : ~{_estimate_minutes(args, dimensions)} min")
    print("=" * 74)

    evaluator = Evaluator(args, output_dir, references)
    resume = load_resume(report_path) if args.resume else {}
    if resume:
        print(f"  resume: {len(resume)} candidate(s) available\n")

    winner = dict(baseline)
    interrupted = False
    try:
        if sweep_ref:
            sweep_reference(evaluator, winner, quick=args.quick, resume=resume)
        for dimension in dimensions:
            winner = sweep_dimension(
                evaluator,
                winner,
                dimension,
                quick=args.quick,
                resume=resume,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[tune] interrupted; the partial report has been saved")
    finally:
        if not args.keep_running:
            print("\n[tune] restoring the original .env configuration ...")
            _restore(original_config)

    if interrupted:
        write_report(report_path, evaluator, args, None, evaluator.active)
        return 130

    feasible = [r for r in evaluator.records if r.feasible and r.score is not None]
    feasible.sort(key=lambda r: float(r.score or 0.0))
    best: Optional[dict[str, str]] = feasible[0].config if feasible else None
    write_report(report_path, evaluator, args, best, evaluator.active)
    _print_summary(evaluator, feasible, best, report_path, tuned_env_path, args)
    print_variant_table(evaluator, args)

    if best:
        # best_config.json is the service-facing artefact; tuned.env is the
        # human-readable one. Both are written so a dry run is still useful.
        write_best_config(
            best_config_path, best, evaluator, args, evaluator.active
        )
        print(f"\n[tune] profile: {best_config_path}")
        print("[tune] app/config.py reads it at the next container start.")

    if best and args.apply:
        keys = [str(d["env"]) for d in SERVER_DIMENSIONS + REQUEST_DIMENSIONS]
        subset = {k: v for k, v in best.items() if k in keys}
        backup = apply_to_env(subset, ENV_PATH, keys)
        print(f"\n[tune] applied to {ENV_PATH} (backup: {backup.name})")
        print("[tune] run: docker compose up -d --force-recreate")
    elif best:
        write_tuned_env(tuned_env_path, best, evaluator, args, evaluator.active)
        print(f"\n[tune] dry run - nothing changed. Tuned block: {tuned_env_path}")
        print("[tune] re-run with --apply to install it.")
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())

