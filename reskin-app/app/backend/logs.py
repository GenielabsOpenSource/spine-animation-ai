"""Visual pipeline logger.

Records per-operation events with snapshotted input/output images so the
frontend can render a debug timeline. Storage layout:

    .genie/logs/
        events.jsonl                    one JSON record per line
        snapshots/{ts}_{label}.png      copies of input/output images

Events are append-only; on each `record()` call we copy the source images
into `snapshots/` (keyed by a millisecond timestamp + label) so that even
when the original file is overwritten in-place by a later operation the
log still has the exact pixels that were seen.
"""
from __future__ import annotations

import io
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image


_MAX_THUMB_DIM = 1024  # downscale snapshots so the modal stays responsive


class PipelineLogger:
    def __init__(self, project_path: Path, workdir: Path):
        self.project_path = Path(project_path)
        self.logs_dir = Path(workdir) / "logs"
        self.snapshots_dir = self.logs_dir / "snapshots"
        self.events_file = self.logs_dir / "events.jsonl"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, src: Path | Image.Image, label: str) -> Optional[Path]:
        """Copy or save `src` to a stable, timestamped path under
        `.genie/logs/snapshots/`. Returns the new path (or None on failure).

        Accepts either a filesystem path or a PIL Image (so callers can log
        in-memory intermediates without writing to disk first)."""
        ts = int(time.time() * 1000)
        # Sanitize label so it's filesystem-safe.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)[:80]
        out = self.snapshots_dir / f"{ts}_{uuid.uuid4().hex[:6]}_{safe}.png"
        try:
            if isinstance(src, Image.Image):
                img = src.convert("RGBA")
            else:
                p = Path(src)
                if not p.is_file():
                    return None
                img = Image.open(p).convert("RGBA")
            # Downscale for quick UI thumbs (preserve aspect).
            w, h = img.size
            longest = max(w, h)
            if longest > _MAX_THUMB_DIM:
                ratio = _MAX_THUMB_DIM / longest
                img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.LANCZOS)
            img.save(out, format="PNG", optimize=True)
            return out
        except Exception as e:
            print(f"[logs] snapshot failed for {label}: {e}", flush=True)
            return None

    def record(
        self,
        operation: str,
        *,
        skin_name: Optional[str] = None,
        slot: Optional[str] = None,
        params: Optional[dict] = None,
        input_paths: Iterable[Optional[Path]] = (),
        output_paths: Iterable[Optional[Path]] = (),
        duration_ms: Optional[float] = None,
        status: str = "ok",
        error: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """Append an event. Paths should be already-snapshotted (call
        `snapshot()` first) so they don't get overwritten by later ops."""
        def _rel(p: Optional[Path]) -> Optional[str]:
            if p is None:
                return None
            try:
                return str(Path(p).relative_to(self.project_path))
            except Exception:
                return str(p)

        event = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "operation": operation,
            "skin_name": skin_name,
            "slot": slot,
            "params": _coerce_params(params or {}),
            "input_images": [_rel(p) for p in input_paths if p is not None],
            "output_images": [_rel(p) for p in output_paths if p is not None],
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
            "notes": notes,
        }
        try:
            with self.events_file.open("a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[logs] write failed: {e}", flush=True)
        return event

    def read_events(self, limit: int = 500) -> list[dict]:
        if not self.events_file.is_file():
            return []
        events: list[dict] = []
        try:
            for line in self.events_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        except Exception:
            return []
        # Newest first.
        return events[-limit:][::-1]

    def clear(self) -> int:
        """Delete all recorded events + snapshots. Returns count cleared."""
        n = 0
        if self.events_file.is_file():
            n = sum(1 for _ in self.events_file.open())
            self.events_file.unlink()
        if self.snapshots_dir.is_dir():
            shutil.rmtree(self.snapshots_dir, ignore_errors=True)
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        return n


def _coerce_params(params: dict) -> dict:
    """Make params JSON-safe (truncate long strings, stringify Paths)."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, str) and len(v) > 4000:
            out[k] = v[:4000] + "…"
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            out[k] = v
        else:
            out[k] = repr(v)
    return out
