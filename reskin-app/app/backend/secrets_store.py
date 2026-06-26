"""User-provided API keys & secret endpoints.

Secrets are stored in ~/.genie-reskin/secrets.json (the user's config dir,
never in the repo). Precedence: a value saved there overrides the matching
env var; if it isn't saved, the env var (e.g. from app/.env) *seeds* the
field. `apply_to_env()` pushes the effective values into os.environ so the
provider modules (gemini.py, bria.py, the SAM endpoints) — which read
os.environ — keep working unchanged.

Nothing here is logged. The SAM server URL is the team's own infrastructure;
it is treated as a secret like the API keys (kept out of the repo, only ever
read from the user's local env / config).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


SECRETS_PATH = Path.home() / ".genie-reskin" / "secrets.json"


@dataclass(frozen=True)
class SecretDef:
    name: str            # env var name
    label: str
    required: bool       # required → missing-key badge + blocks Generate/Retouch
    kind: str            # "key" | "url"
    help_url: str = ""
    description: str = ""


# The single source of truth for which secrets the app surfaces.
SECRET_DEFS: tuple[SecretDef, ...] = (
    SecretDef(
        name="GEMINI_API_KEY",
        label="Gemini API key",
        required=True,
        kind="key",
        help_url="https://aistudio.google.com/app/apikey",
        description="Powers every reskin and Retouch. Without it nothing generates.",
    ),
    SecretDef(
        name="FAL_KEY",
        label="fal.ai key",
        required=True,
        kind="key",
        help_url="https://fal.ai/dashboard/keys",
        description="Runs Bria background removal — clean part edges during rebake and after Retouch.",
    ),
    SecretDef(
        name="SAM_SERVER_URL",
        label="SAM server URL",
        required=False,
        kind="url",
        help_url="",
        description="Your own SAM segmentation server. Optional — only the SAM segmentation method and Magic-select use it.",
    ),
)

_BY_NAME = {d.name: d for d in SECRET_DEFS}

# Snapshot of env-provided values at import time (after load_dotenv ran in
# server.py). This is the "seed" a field falls back to when there's no saved
# override — so clearing a saved value reverts to the .env value, not nothing.
_ENV_SEED = {d.name: os.environ.get(d.name, "") for d in SECRET_DEFS}


def _load_saved() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        raw = json.loads(SECRETS_PATH.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if isinstance(k, str) and k in _BY_NAME}


def effective(name: str) -> str:
    """Saved value wins; otherwise the .env seed fills it in."""
    saved = _load_saved().get(name)
    if saved:
        return saved
    return _ENV_SEED.get(name, "")


def _sync_env() -> None:
    """Make os.environ reflect the effective values so provider modules that
    read os.environ (gemini.py, bria.py, the SAM endpoints) stay in sync."""
    for d in SECRET_DEFS:
        val = effective(d.name)
        if val:
            os.environ[d.name] = val
        else:
            os.environ.pop(d.name, None)


def apply_to_env() -> None:
    """Push effective secrets into os.environ (saved overrides take precedence
    over the .env). Call once at startup; save() re-syncs after edits."""
    _sync_env()


def save(updates: dict) -> None:
    """Merge `updates` ({ENV_NAME: value}) into the saved store and the live
    process env. An empty/blank value removes the override (the env var, if
    any, re-seeds the field). The file is written user-only (0600)."""
    saved = _load_saved()
    for name, val in updates.items():
        if name not in _BY_NAME:
            continue
        text = "" if val is None else str(val).strip()
        if text:
            saved[name] = text
        else:
            saved.pop(name, None)
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_PATH.write_text(json.dumps(saved, indent=2))
    try:
        os.chmod(SECRETS_PATH, 0o600)
    except OSError:
        pass
    _sync_env()


def status_payload() -> list[dict]:
    """Shape for the Settings UI. Includes the effective value (the user opted
    to show saved values) plus whether it's set and whether it came from the
    .env seed vs an explicit save."""
    saved = _load_saved()
    out: list[dict] = []
    for d in SECRET_DEFS:
        eff = effective(d.name)
        out.append({
            "name": d.name,
            "label": d.label,
            "required": d.required,
            "kind": d.kind,
            "help_url": d.help_url,
            "description": d.description,
            "value": eff,
            "is_set": bool(eff),
            "from_env": d.name not in saved and bool(_ENV_SEED.get(d.name)),
        })
    return out
