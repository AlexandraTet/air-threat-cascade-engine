"""Lightweight, crash-proof configuration loader.

Reads settings from the process environment, optionally pre-populated from a
``.env`` file, and exposes them as an immutable :class:`Settings` object.
Every value has a safe default so the application always starts: a missing
``GEMINI_API_KEY`` simply switches the LLM layer off (offline / synthetic
mode) instead of raising.

Design rules
------------
* **No hard dependency** on ``python-dotenv``; a small built-in parser handles
  the common ``KEY=value`` / ``KEY="quoted value"`` / ``export KEY=value``
  forms and ``#`` comments.
* **Real environment wins** over ``.env`` by default (12-factor behaviour),
  so container or CI variables are never silently overridden by a stale file.
* **Invalid values never crash**: a non-numeric ``SYNTHETIC_SEED`` or an
  unknown timezone is logged as a warning and replaced with the default.
* **Secrets never leak into logs**: :meth:`Settings.masked_key` and
  :meth:`Settings.as_dict` redact the API key.

Example
-------
>>> from src.config import get_settings
>>> settings = get_settings()
>>> if settings.llm_enabled:
...     client = make_gemini_client(settings.gemini_api_key)
... else:
...     client = None  # offline mode; rule-based summaries only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, Final, Mapping, MutableMapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "DEFAULTS",
    "Settings",
    "get_settings",
    "load_settings",
    "parse_dotenv",
    "load_dotenv_file",
    "reset_settings_cache",
]

logger = logging.getLogger(__name__)

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on", "enabled"})
_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off", "disabled", ""})

DEFAULTS: Final[Mapping[str, str]] = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-2.5-flash",
    "LLM_ENABLED": "true",
    "ALARMS_CSV_PATH": "data/alarms.csv",
    "SYNTHETIC_FALLBACK": "true",
    "SYNTHETIC_RECORDS": "2000",
    "SYNTHETIC_SEED": "42",
    "SYNTHETIC_START_DATE": "2026-01-01",
    "DEFAULT_CUTOFF_TIMEZONE": "UTC",
    "ACTIVE_WINDOW_HOURS": "8",
    "CASCADE_MAX_LAG_MIN": "90",
    "CASCADE_DECAY_MIN": "60",
    "APP_ENV": "development",
    "LOG_LEVEL": "INFO",
}
"""Default raw values, mirrored in ``.env.example``."""


# --------------------------------------------------------------------------- #
# .env parsing
# --------------------------------------------------------------------------- #
def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env``-style text into a mapping.

    Supported syntax: blank lines, ``#`` comments, ``KEY=value``,
    ``export KEY=value``, single- or double-quoted values (quotes stripped,
    ``\\n`` and ``\\t`` unescaped inside double quotes) and inline comments
    after unquoted values. Malformed lines are skipped with a warning.

    Args:
        text: Raw file contents.

    Returns:
        Ordered mapping of variable names to string values.
    """
    result: dict[str, str] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            logger.warning(".env line %d ignored (no '='): %r", line_no, raw_line)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            logger.warning(".env line %d ignored (invalid name): %r", line_no, raw_line)
            continue
        value = value.strip()
        if value and value[0] in {'"', "'"}:
            quote = value[0]
            closing = value.find(quote, 1)
            if closing != -1:
                # Quoted value; anything after the closing quote (e.g. an
                # inline comment) is discarded.
                value = value[1:closing]
                if quote == '"':
                    value = value.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            else:
                logger.warning(".env line %d has an unterminated quote; value kept verbatim", line_no)
        else:
            hash_index = value.find(" #")
            if hash_index != -1:
                value = value[:hash_index].rstrip()
        result[key] = value
    return result


def load_dotenv_file(
    path: os.PathLike[str] | str = ".env",
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load a ``.env`` file into an environment mapping if it exists.

    Args:
        path: Path to the file. A missing file is not an error.
        environ: Mapping to populate; defaults to :data:`os.environ`.
        override: When ``False`` (default) variables already present in
            ``environ`` are left untouched.

    Returns:
        The variables that were actually written.
    """
    target = os.environ if environ is None else environ
    file_path = Path(path)
    if not file_path.is_file():
        logger.debug("No .env file at %s; using process environment only", file_path)
        return {}
    try:
        parsed = parse_dotenv(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s (%s); continuing without it", file_path, exc)
        return {}
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in target:
            target[key] = value
            applied[key] = value
    logger.debug("Loaded %d variable(s) from %s", len(applied), file_path)
    return applied


# --------------------------------------------------------------------------- #
# Typed coercion helpers (never raise)
# --------------------------------------------------------------------------- #
def _raw(env: Mapping[str, str], key: str) -> str:
    """Return the environment value or the documented default."""
    value = env.get(key)
    return DEFAULTS[key] if value is None else value


def _as_str(env: Mapping[str, str], key: str) -> str:
    value = _raw(env, key).strip()
    return value if value else DEFAULTS[key]


def _as_bool(env: Mapping[str, str], key: str) -> bool:
    value = _raw(env, key).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logger.warning("%s=%r is not a boolean; using default %r", key, value, DEFAULTS[key])
    return DEFAULTS[key].lower() in _TRUE_VALUES


def _as_int(env: Mapping[str, str], key: str, *, minimum: Optional[int] = None) -> int:
    value = _raw(env, key).strip()
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %s", key, value, DEFAULTS[key])
        parsed = int(DEFAULTS[key])
    if minimum is not None and parsed < minimum:
        logger.warning("%s=%d is below the minimum %d; using default %s", key, parsed, minimum, DEFAULTS[key])
        parsed = int(DEFAULTS[key])
    return parsed


def _as_float(env: Mapping[str, str], key: str, *, minimum: Optional[float] = None) -> float:
    value = _raw(env, key).strip()
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("%s=%r is not a number; using default %s", key, value, DEFAULTS[key])
        parsed = float(DEFAULTS[key])
    if minimum is not None and parsed < minimum:
        logger.warning("%s=%s is below the minimum %s; using default %s", key, parsed, minimum, DEFAULTS[key])
        parsed = float(DEFAULTS[key])
    return parsed


def _as_timezone(env: Mapping[str, str], key: str) -> str:
    value = _as_str(env, key)
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("%s=%r is not a known IANA timezone; using UTC", key, value)
        return "UTC"
    return value


def _as_iso_date(env: Mapping[str, str], key: str) -> str:
    value = _as_str(env, key)
    try:
        date.fromisoformat(value)
    except ValueError:
        logger.warning("%s=%r is not an ISO date (YYYY-MM-DD); using default %s", key, value, DEFAULTS[key])
        return DEFAULTS[key]
    return value


def _as_log_level(env: Mapping[str, str], key: str) -> str:
    value = _as_str(env, key).upper()
    if value not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        logger.warning("%s=%r is not a logging level; using INFO", key, value)
        return "INFO"
    return value


# --------------------------------------------------------------------------- #
# Settings object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings.

    Attributes:
        gemini_api_key: Gemini API key, or ``None`` when absent/blank.
        gemini_model: Gemini model identifier for narrative summaries.
        llm_requested: ``LLM_ENABLED`` flag as configured.
        alarms_csv_path: Historical alarm CSV path.
        synthetic_fallback: Whether to synthesise data when the CSV is missing.
        synthetic_records: Rows generated in synthetic mode.
        synthetic_seed: Random seed for synthetic mode.
        synthetic_start_date: First synthetic alarm date (ISO).
        default_cutoff_timezone: IANA zone for interpreting naive cutoffs.
        active_window_hours: Staleness window for open-ended alarms.
        cascade_max_lag_min: Default propagation lag window (minutes).
        cascade_decay_min: Default trigger-influence decay (minutes).
        app_env: Deployment environment label.
        log_level: Logging level name.
        warnings: Non-fatal configuration issues found while loading.
    """

    gemini_api_key: Optional[str] = None
    gemini_model: str = DEFAULTS["GEMINI_MODEL"]
    llm_requested: bool = True
    alarms_csv_path: str = DEFAULTS["ALARMS_CSV_PATH"]
    synthetic_fallback: bool = True
    synthetic_records: int = int(DEFAULTS["SYNTHETIC_RECORDS"])
    synthetic_seed: int = int(DEFAULTS["SYNTHETIC_SEED"])
    synthetic_start_date: str = DEFAULTS["SYNTHETIC_START_DATE"]
    default_cutoff_timezone: str = DEFAULTS["DEFAULT_CUTOFF_TIMEZONE"]
    active_window_hours: float = float(DEFAULTS["ACTIVE_WINDOW_HOURS"])
    cascade_max_lag_min: float = float(DEFAULTS["CASCADE_MAX_LAG_MIN"])
    cascade_decay_min: float = float(DEFAULTS["CASCADE_DECAY_MIN"])
    app_env: str = DEFAULTS["APP_ENV"]
    log_level: str = DEFAULTS["LOG_LEVEL"]
    warnings: tuple[str, ...] = field(default=(), compare=False)

    # ---- derived flags -------------------------------------------------- #
    @property
    def has_gemini_key(self) -> bool:
        """``True`` when a non-blank ``GEMINI_API_KEY`` was supplied."""
        return bool(self.gemini_api_key)

    @property
    def llm_enabled(self) -> bool:
        """``True`` only when LLM use is requested *and* a key is present."""
        return self.llm_requested and self.has_gemini_key

    @property
    def offline_mode(self) -> bool:
        """``True`` when no LLM calls will be made (missing key or disabled)."""
        return not self.llm_enabled

    @property
    def csv_available(self) -> bool:
        """``True`` when the configured alarm CSV exists on disk."""
        return Path(self.alarms_csv_path).is_file()

    @property
    def cutoff_tzinfo(self) -> ZoneInfo:
        """The configured timezone as a :class:`zoneinfo.ZoneInfo`."""
        return ZoneInfo(self.default_cutoff_timezone)

    # ---- presentation ----------------------------------------------------- #
    def masked_key(self) -> str:
        """Redacted key for logs and UIs (``"AIza…9xQ2"`` or ``"<not set>"``)."""
        key = self.gemini_api_key or ""
        if not key:
            return "<not set>"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}…{key[-4:]}"

    def as_dict(self) -> dict[str, Any]:
        """Settings as a plain dict with the API key redacted."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["gemini_api_key"] = self.masked_key()
        data["llm_enabled"] = self.llm_enabled
        data["offline_mode"] = self.offline_mode
        return data

    def describe(self) -> str:
        """One-line human-readable summary suitable for a startup log."""
        mode = "LLM online" if self.llm_enabled else "offline / rule-based"
        source = "csv" if self.csv_available else (
            "synthetic fallback" if self.synthetic_fallback else "csv (missing!)"
        )
        return (
            f"env={self.app_env} mode={mode} model={self.gemini_model} key={self.masked_key()} "
            f"data={source} tz={self.default_cutoff_timezone}"
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class _WarningCollector(logging.Handler):
    """Capture warnings emitted while coercing values into ``Settings.warnings``."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def load_settings(
    env_file: os.PathLike[str] | str | None = ".env",
    *,
    environ: Mapping[str, str] | None = None,
    override: bool = False,
) -> Settings:
    """Build a :class:`Settings` instance without caching.

    Args:
        env_file: ``.env`` path to merge in first; ``None`` skips file loading.
        environ: Environment mapping to read from. Defaults to
            :data:`os.environ` (which the ``.env`` file may populate). Passing
            an explicit mapping keeps the process environment untouched,
            which is what tests want.
        override: Let ``.env`` values override existing variables.

    Returns:
        A fully validated :class:`Settings`. Never raises for missing or
        malformed values; problems are logged and listed in ``warnings``.
    """
    env: MutableMapping[str, str]
    if environ is None:
        env = os.environ
    else:
        env = dict(environ)
    if env_file is not None:
        load_dotenv_file(env_file, environ=env, override=override)

    collector = _WarningCollector()
    logger.addHandler(collector)
    try:
        key = _raw(env, "GEMINI_API_KEY").strip() or None
        settings = Settings(
            gemini_api_key=key,
            gemini_model=_as_str(env, "GEMINI_MODEL"),
            llm_requested=_as_bool(env, "LLM_ENABLED"),
            alarms_csv_path=_as_str(env, "ALARMS_CSV_PATH"),
            synthetic_fallback=_as_bool(env, "SYNTHETIC_FALLBACK"),
            synthetic_records=_as_int(env, "SYNTHETIC_RECORDS", minimum=1),
            synthetic_seed=_as_int(env, "SYNTHETIC_SEED", minimum=0),
            synthetic_start_date=_as_iso_date(env, "SYNTHETIC_START_DATE"),
            default_cutoff_timezone=_as_timezone(env, "DEFAULT_CUTOFF_TIMEZONE"),
            active_window_hours=_as_float(env, "ACTIVE_WINDOW_HOURS", minimum=0.0),
            cascade_max_lag_min=_as_float(env, "CASCADE_MAX_LAG_MIN", minimum=1.0),
            cascade_decay_min=_as_float(env, "CASCADE_DECAY_MIN", minimum=1.0),
            app_env=_as_str(env, "APP_ENV").lower(),
            log_level=_as_log_level(env, "LOG_LEVEL"),
        )
    finally:
        logger.removeHandler(collector)

    notes = list(collector.messages)
    if not settings.has_gemini_key:
        notes.append("GEMINI_API_KEY is not set; running in offline mode (LLM features disabled).")
    elif not settings.llm_requested:
        notes.append("LLM_ENABLED is false; Gemini key present but LLM features are disabled.")
    if not settings.csv_available and not settings.synthetic_fallback:
        notes.append(
            f"ALARMS_CSV_PATH '{settings.alarms_csv_path}' does not exist and SYNTHETIC_FALLBACK is "
            "false; data loading will fail until a CSV is provided."
        )
    for note in notes:
        logger.info("config: %s", note)
    return Settings(**{**{f.name: getattr(settings, f.name) for f in fields(settings)},
                       "warnings": tuple(notes)})


_CACHE: dict[str, Settings] = {}


def get_settings(env_file: os.PathLike[str] | str | None = ".env", *, reload: bool = False) -> Settings:
    """Return the process-wide cached :class:`Settings`.

    Args:
        env_file: ``.env`` path used on first load (or when ``reload`` is set).
        reload: Force re-reading the environment and file.

    Returns:
        The cached :class:`Settings` instance.
    """
    cache_key = str(env_file)
    if reload or cache_key not in _CACHE:
        _CACHE[cache_key] = load_settings(env_file)
    return _CACHE[cache_key]


def reset_settings_cache() -> None:
    """Clear the cached settings (useful in tests)."""
    _CACHE.clear()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    current = get_settings()
    print(current.describe())
    for item in current.warnings:
        print(" -", item)
