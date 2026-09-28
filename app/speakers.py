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


@dataclass
class SpeakerRecord:
    """One registered voice."""

    id: str
    audio_path: str
    text_path: str
    reference_text: str
    audio_bytes: int
    audio_filename: str
    created: str
    requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        )
        self._write_meta(record)
        logger.info("Registered speaker %s (%d bytes)", identifier, len(audio_bytes))
        return record

    def get(self, speaker_id: str) -> Optional[SpeakerRecord]:
        meta = self._meta_path(speaker_id)
        if not meta.exists():
            return None
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("speaker %s has unreadable metadata: %s", speaker_id, exc)
            return None
        record = SpeakerRecord(
            id=str(payload.get("id") or speaker_id),
            audio_path=str(payload.get("audio_path") or ""),
            text_path=str(payload.get("text_path") or ""),
            reference_text=str(payload.get("reference_text") or ""),
            audio_bytes=int(payload.get("audio_bytes") or 0),
            audio_filename=str(payload.get("audio_filename") or "reference.wav"),
            created=str(payload.get("created") or ""),
            requests=int(payload.get("requests") or 0),
        )
        # A clip deleted behind our back must not resolve as usable.
        if not record.audio_path or not Path(record.audio_path).exists():
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

