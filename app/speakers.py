"""Speaker registry: register a voice once, reuse it for every request (v3 Task 3.3).

Why a registry at all
---------------------
``dots_tts`` already caches the *latents* it derives from a reference clip
(``DotsTtsModel._prompt_feature_cache``, keyed by a hash of the audio), so
repeated requests with the same clip skip prompt encoding internally. What that
does **not** avoid is the client re-uploading a ~1 MB WAV and re-sending the
transcript on every chunk of a long-form job.

A registry moves that cost to registration time: the clip is written once, and
each subsequent request carries a ~20-byte ``speaker_id`` instead of a
megabyte of audio. For the 7-chunk pipeline run this removes 6 of 7 uploads.

Storage is on disk rather than in RAM on purpose — see
``PathSettings.speakers_dir``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import LOG_NAMESPACE

logger = logging.getLogger(f"{LOG_NAMESPACE}.api")

__all__ = ["SpeakerRecord", "SpeakerRegistry"]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _optional_bool(value: Any) -> Optional[bool]:
    """Read a consent flag that may be absent, and must not be guessed.

    ``None`` (unknown) is preserved rather than coerced to ``False``: "we never
    asked" and "we were told no" are different answers, and collapsing them
    would make a registry that predates consent look like a registry of
    unconsented voices.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "on"}:
            return True
        if normalised in {"0", "false", "no", "n", "off"}:
            return False
    return None


@dataclass
class SpeakerRecord:
    """One registered voice.

    The ``consent_*`` fields exist because a voice clone is a likeness, not a
    file. Recording who acknowledged that, and when, means the registry can
    later answer "do we have permission for this voice?" - which is the first
    question anyone should have to ask before using a stored clip, and the one
    an abuse report will ask.

    They are stored per registration rather than in a separate consent ledger
    because a ledger is a second thing to keep consistent with the data it
    governs; here the answer is in the same file as the voice.
    """

    id: str
    audio_path: str
    text_path: str
    reference_text: str
    audio_bytes: int
    audio_filename: str
    created: str
    requests: int = 0
    #: ``None`` on records written before consent was tracked, which is *not* the
    #: same as ``False``. ``None`` means "unknown"; the difference matters when
    #: someone asks whether a specific clip was registered with consent.
    consent_acknowledged: Optional[bool] = None
    consent_at: str = ""
    #: Free text: who owns the voice, and on what basis it may be used.
    consent_owner: str = ""
    consent_basis: str = ""

    def as_dict(self) -> dict[str, Any]:
        """The flat form written to the sidecar.

        Used for persistence, so it stays exactly the set of stored fields -
        no derived keys, or the JSON on disk and the dataclass drift apart.
        """
        return asdict(self)

    def as_api_dict(self) -> dict[str, Any]:
        """The API form, with the consent answer as one nested object.

        Nested so a client does not have to know that ``null`` (never asked)
        and ``false`` (asked, refused) are different states - the distinction
        the whole field exists to preserve.
        """
        return {
            **self.as_dict(),
            "consent": {
                "status": self.consent_status,
                "acknowledged": self.consent_acknowledged,
                "recorded_at": self.consent_at,
                "owner": self.consent_owner,
                "basis": self.consent_basis,
            },
        }

    @property
    def consent_status(self) -> str:
        """``granted`` / ``not_acknowledged`` / ``unknown`` for the API and UI."""
        if self.consent_acknowledged is None:
            return "unknown"
        return "granted" if self.consent_acknowledged else "not_acknowledged"


class SpeakerRegistry:
    """Filesystem-backed speaker store for a single uvicorn worker."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- internals ---------------------------------------------------------- #
    def _paths(self, speaker_id: str) -> tuple[Path, Path]:
        return self.root / f"{speaker_id}.wav", self.root / f"{speaker_id}.txt"

    def _meta_path(self, speaker_id: str) -> Path:
        return self.root / f"{speaker_id}.json"

    @staticmethod
    def validate_id(speaker_id: str) -> str:
        candidate = (speaker_id or "").strip()
        if not _ID_PATTERN.match(candidate):
            raise ValueError(
                "speaker_id must be 1-64 characters of letters, digits, '.', '_' "
                "or '-', and start with a letter or digit"
            )
        return candidate

    def _write_meta(self, record: SpeakerRecord) -> None:
        self._meta_path(record.id).write_text(
            json.dumps(record.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _auto_id(self) -> str:
        """First free ``speaker-N``; ids stay short and predictable."""
        index = 1
        while (self.root / f"speaker-{index}.wav").exists():
            index += 1
        return f"speaker-{index}"

    # -- public API --------------------------------------------------------- #
    def register(
        self,
        *,
        speaker_id: Optional[str],
        audio_bytes: bytes,
        audio_filename: str,
        reference_text: str,
        overwrite: bool = False,
        consent_acknowledged: Optional[bool] = None,
        consent_owner: str = "",
        consent_basis: str = "",
    ) -> SpeakerRecord:
        """Persist a voice. Raises ``ValueError`` on bad input or a name clash."""
        if not audio_bytes:
            raise ValueError("'reference_audio' is empty.")
        text = (reference_text or "").strip()
        if not text:
            raise ValueError(
                "'reference_text' must not be empty: continuation voice cloning "
                "needs the exact transcript of the reference audio."
            )
        identifier = self.validate_id(speaker_id) if speaker_id else self._auto_id()
        audio_path, text_path = self._paths(identifier)
        if audio_path.exists() and not overwrite:
            raise ValueError(
                f"speaker_id {identifier!r} already exists; pass overwrite=true "
                "to replace it"
            )
        # Stage then rename, so a crash cannot leave a truncated clip registered
        # under a valid id.
        staging = audio_path.with_suffix(".wav.tmp")
        staging.write_bytes(audio_bytes)
        staging.replace(audio_path)
        text_path.write_text(text, encoding="utf-8")

        record = SpeakerRecord(
            id=identifier,
            audio_path=str(audio_path),
            text_path=str(text_path),
            reference_text=text,
            audio_bytes=len(audio_bytes),
            audio_filename=audio_filename or "reference.wav",
            created=_now(),
            # Recorded on *every* write, including an overwrite, so the timestamp
            # always describes the audio currently on disk rather than whatever
            # was registered first under that id.
            consent_acknowledged=consent_acknowledged,
            consent_at=_now() if consent_acknowledged else "",
            consent_owner=(consent_owner or "").strip(),
            consent_basis=(consent_basis or "").strip(),
        )
        self._write_meta(record)
        logger.info("Registered speaker %s (%d bytes)", identifier, len(audio_bytes))
        return record

    def _resolve_within_root(self, candidate: str, speaker_id: str) -> Optional[Path]:
        """Resolve a path from the metadata, refusing anything outside the root.

        The metadata sidecar is a plain JSON file inside a **host bind mount**
        (``./data``), so it is not fully trusted input: anything able to write
        ``data/speakers/<id>.json`` can put an arbitrary ``audio_path`` in it,
        and callers such as ``POST /v1/audio/speech`` then use that path as the
        synthesis prompt. Requiring the resolved path to sit inside the speakers
        directory means a planted sidecar can only ever point at another
        registered clip, never at ``/etc/passwd`` or a host file.
        """
        try:
            resolved = Path(candidate).resolve()
            root = self.root.resolve()
        except (OSError, ValueError):
            return None
        if resolved != root and root not in resolved.parents:
            logger.warning(
                "speaker %s: audio_path %s is outside %s; refusing",
                speaker_id,
                candidate,
                root,
            )
            return None
        return resolved

    def get(self, speaker_id: str) -> Optional[SpeakerRecord]:
        meta = self._meta_path(speaker_id)
        if not meta.exists():
            return None
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("speaker %s has unreadable metadata: %s", speaker_id, exc)
            return None
        audio_path = self._resolve_within_root(
            str(payload.get("audio_path") or ""), speaker_id
        )
        if audio_path is None:
            return None
        record = SpeakerRecord(
            id=str(payload.get("id") or speaker_id),
            audio_path=str(audio_path),
            text_path=str(payload.get("text_path") or ""),
            reference_text=str(payload.get("reference_text") or ""),
            audio_bytes=int(payload.get("audio_bytes") or 0),
            audio_filename=str(payload.get("audio_filename") or "reference.wav"),
            created=str(payload.get("created") or ""),
            requests=int(payload.get("requests") or 0),
            # Sidecars written before consent tracking have no such key, and
            # they must keep loading: a missing field is "unknown", never
            # "refused", or upgrading the service would orphan real voices.
            consent_acknowledged=_optional_bool(payload.get("consent_acknowledged")),
            consent_at=str(payload.get("consent_at") or ""),
            consent_owner=str(payload.get("consent_owner") or ""),
            consent_basis=str(payload.get("consent_basis") or ""),
        )
        # A clip deleted behind our back must not resolve as usable.
        if not audio_path.exists():
            logger.warning("speaker %s: audio file missing on disk", speaker_id)
            return None
        return record

    def list(self) -> list[SpeakerRecord]:
        records = []
        for meta in sorted(self.root.glob("*.json")):
            record = self.get(meta.stem)
            if record is not None:
                records.append(record)
        return records

    def delete(self, speaker_id: str) -> bool:
        meta = self._meta_path(speaker_id)
        if not meta.exists():
            return False
        for path in (*self._paths(speaker_id), meta):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - permissions
                logger.warning("could not remove %s: %s", path, exc)
        logger.info("Deleted speaker %s", speaker_id)
        return True

    def count_request(self, speaker_id: str) -> None:
        """Bump the reuse counter so cache effectiveness is visible in /health."""
        record = self.get(speaker_id)
        if record is None:
            return
        record.requests += 1
        try:
            self._write_meta(record)
        except OSError:  # pragma: no cover - best-effort telemetry
            pass

    def clear(self) -> int:
        """Remove every registered speaker. Returns how many were deleted."""
        removed = 0
        for meta in list(self.root.glob("*.json")):
            if self.delete(meta.stem):
                removed += 1
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        return removed

