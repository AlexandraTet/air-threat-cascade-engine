"""Historical air-raid alarm ingestion for the Air-Threat Cascade engine.

This module is the single entry point for turning raw alarm logs into a clean,
chronologically ordered :class:`pandas.DataFrame` that downstream feature
builders and models can trust. It has three responsibilities:

1. **Loading & cleaning** real CSV exports (column-name normalisation, datetime
   parsing, missing-value handling, deduplication, chronological sorting).
2. **Strict temporal safeguarding**: every public loading path accepts a
   ``cutoff_time`` ``T`` and guarantees that no record whose ``start_time`` is
   ``>= T`` survives. Optionally, ``end_time`` values that lie at or beyond the
   cutoff are masked, because at time ``T`` the end of a still-running alarm is
   not yet observable.
3. **Synthetic fallback** for offline execution, producing data that mimics
   Ukrainian oblast-level alarms with a mix of fast-wave (ballistic / aviation /
   cruise) and slow-wave (Shahed-type / reconnaissance UAV) threats.

Canonical output schema
-----------------------
=============  =====================  =======================================
Column         dtype                  Meaning
=============  =====================  =======================================
alarm_id       int64                  Stable per-row identifier.
region         str                    Ukrainian oblast / city name.
start_time     datetime64[ns, UTC]    Alarm activation time.
end_time       datetime64[ns, UTC]    Alarm deactivation time (NaT if unknown).
threat_type    str                    One of :data:`THREAT_TYPES` or
                                      ``"unknown"``.
wave_class     str                    ``"fast"``, ``"slow"`` or ``"unknown"``.
duration_min   float64                ``end_time - start_time`` in minutes.
=============  =====================  =======================================

Example
-------
>>> from datetime import datetime, timezone
>>> from src.loader import AirThreatLoader
>>> loader = AirThreatLoader()
>>> cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
>>> df = loader.load_or_generate("data/alarms.csv", cutoff_time=cutoff)
>>> assert (df["start_time"] < pd.Timestamp(cutoff)).all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "AirThreatLoader",
    "LoaderError",
    "SchemaValidationError",
    "TemporalLeakageError",
    "ThreatWave",
    "THREAT_TYPES",
    "UKRAINIAN_REGIONS",
    "CANONICAL_COLUMNS",
    "classify_threat_wave",
    "normalize_threat_type",
    "to_utc_timestamp",
    "generate_synthetic_alarms",
    "load_and_filter_alarms",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
TimeLike = Union[datetime, pd.Timestamp, str]


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
class ThreatWave(str, Enum):
    """Kinematic class of an air threat.

    ``FAST`` covers short-warning threats (ballistic and cruise missiles,
    tactical aviation, guided bombs) where regional alarms cascade within
    minutes. ``SLOW`` covers loitering munitions and UAVs (Shahed-type) that
    cross the country over several hours, producing long, overlapping alarm
    chains.
    """

    FAST = "fast"
    SLOW = "slow"
    UNKNOWN = "unknown"


THREAT_TYPES: Final[Mapping[str, ThreatWave]] = {
    "ballistic_missile": ThreatWave.FAST,
    "cruise_missile": ThreatWave.FAST,
    "aviation": ThreatWave.FAST,
    "guided_bomb": ThreatWave.FAST,
    "artillery": ThreatWave.FAST,
    "uav_shahed": ThreatWave.SLOW,
    "uav_recon": ThreatWave.SLOW,
}
"""Known threat labels mapped to their kinematic wave class."""

# Free-form label fragments -> canonical threat key. Order matters for the
# substring fallback in :func:`normalize_threat_type` (more specific first).
_THREAT_ALIASES: Final[Mapping[str, str]] = {
    "ballistic": "ballistic_missile",
    "iskander": "ballistic_missile",
    "kinzhal": "ballistic_missile",
    "cruise": "cruise_missile",
    "kalibr": "cruise_missile",
    "missile": "cruise_missile",
    "aircraft": "aviation",
    "mig_31": "aviation",
    "air": "aviation",
    "kab": "guided_bomb",
    "shahed": "uav_shahed",
    "geran": "uav_shahed",
    "drone": "uav_shahed",
    "recon": "uav_recon",
    "uav": "uav_shahed",
    "shelling": "artillery",
}

UKRAINIAN_REGIONS: Final[Sequence[str]] = (
    "Kyiv City",
    "Kyiv Oblast",
    "Kharkiv Oblast",
    "Odesa Oblast",
    "Dnipropetrovsk Oblast",
    "Zaporizhzhia Oblast",
    "Mykolaiv Oblast",
    "Kherson Oblast",
    "Sumy Oblast",
    "Chernihiv Oblast",
    "Poltava Oblast",
    "Cherkasy Oblast",
    "Kirovohrad Oblast",
    "Vinnytsia Oblast",
    "Zhytomyr Oblast",
    "Khmelnytskyi Oblast",
    "Rivne Oblast",
    "Volyn Oblast",
    "Lviv Oblast",
    "Ternopil Oblast",
    "Ivano-Frankivsk Oblast",
    "Chernivtsi Oblast",
    "Zakarpattia Oblast",
    "Donetsk Oblast",
    "Luhansk Oblast",
)
"""Oblast-level alarm regions (Crimea excluded: no civil-alert feed)."""

# Relative alarm frequency per region. Front-line and major-hub regions get a
# higher weight so synthetic data reflects the real spatial skew.
_REGION_WEIGHTS: Final[Mapping[str, float]] = {
    "Kyiv City": 2.2,
    "Kyiv Oblast": 2.4,
    "Kharkiv Oblast": 3.5,
    "Odesa Oblast": 2.6,
    "Dnipropetrovsk Oblast": 3.0,
    "Zaporizhzhia Oblast": 2.8,
    "Mykolaiv Oblast": 2.4,
    "Kherson Oblast": 2.6,
    "Sumy Oblast": 2.8,
    "Chernihiv Oblast": 1.8,
    "Poltava Oblast": 1.6,
    "Cherkasy Oblast": 1.3,
    "Kirovohrad Oblast": 1.4,
    "Vinnytsia Oblast": 1.3,
    "Zhytomyr Oblast": 1.3,
    "Khmelnytskyi Oblast": 1.2,
    "Rivne Oblast": 1.0,
    "Volyn Oblast": 0.9,
    "Lviv Oblast": 1.0,
    "Ternopil Oblast": 0.8,
    "Ivano-Frankivsk Oblast": 0.8,
    "Chernivtsi Oblast": 0.6,
    "Zakarpattia Oblast": 0.4,
    "Donetsk Oblast": 2.4,
    "Luhansk Oblast": 0.7,
}

# Threat-type mix *per burst*. Slow-wave bursts span ~6 regions versus ~3 for
# fast-wave bursts, so these burst-level weights yield roughly 60% slow / 40%
# fast at the row level: Shahed-type UAVs dominate the raw alarm count, as in
# the real feed, while fast-wave events stay well represented.
_THREAT_WEIGHTS: Final[Mapping[str, float]] = {
    "uav_shahed": 0.35,
    "uav_recon": 0.08,
    "ballistic_missile": 0.18,
    "cruise_missile": 0.14,
    "aviation": 0.15,
    "guided_bomb": 0.07,
    "artillery": 0.03,
}

# Duration model (log-normal) per threat type, in minutes: (median, sigma).
_DURATION_PARAMS: Final[Mapping[str, tuple[float, float]]] = {
    "ballistic_missile": (14.0, 0.45),
    "cruise_missile": (55.0, 0.55),
    "aviation": (22.0, 0.50),
    "guided_bomb": (30.0, 0.50),
    "artillery": (35.0, 0.60),
    "uav_shahed": (135.0, 0.60),
    "uav_recon": (70.0, 0.65),
}

CANONICAL_COLUMNS: Final[Sequence[str]] = (
    "alarm_id",
    "region",
    "start_time",
    "end_time",
    "threat_type",
    "wave_class",
    "duration_min",
)
"""Ordered output columns produced by every loader method."""

REQUIRED_SOURCE_COLUMNS: Final[Sequence[str]] = ("region", "start_time")
"""Columns that must be present (after alias resolution) in any CSV."""

# Lower-cased source header -> canonical column name. Covers the common export
# formats (alerts.in.ua archives, Telegram scrapes, hand-curated sheets).
DEFAULT_COLUMN_ALIASES: Final[Mapping[str, str]] = {
    # region
    "region": "region",
    "region_title": "region",
    "region_name": "region",
    "oblast": "region",
    "location": "region",
    "area": "region",
    # start
    "start_time": "start_time",
    "start": "start_time",
    "started_at": "start_time",
    "timestamp": "start_time",
    "alarm_start": "start_time",
    "begin": "start_time",
    # end
    "end_time": "end_time",
    "end": "end_time",
    "clean_end": "end_time",
    "finished_at": "end_time",
    "ended_at": "end_time",
    "alarm_end": "end_time",
    "finish": "end_time",
    # threat
    "threat_type": "threat_type",
    "threat": "threat_type",
    "alert_type": "threat_type",
    "alarm_type": "threat_type",
    "type": "threat_type",
    # id
    "alarm_id": "alarm_id",
    "id": "alarm_id",
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class LoaderError(Exception):
    """Base class for all errors raised by this module."""


class SchemaValidationError(LoaderError):
    """Raised when a source file lacks required columns or has unusable data."""


class TemporalLeakageError(LoaderError):
    """Raised when post-filter verification still finds rows at or past cutoff.

    This should be unreachable in practice; it exists as a hard tripwire so
    that any future regression in the filtering logic fails loudly rather than
    silently contaminating a model.
    """


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def normalize_threat_type(threat_type: object) -> str:
    """Normalise a raw threat label to a key of :data:`THREAT_TYPES`.

    Matching is case-insensitive and tolerant of whitespace, hyphens and a
    set of common synonyms (``"Shahed"``, ``"Iskander"``, ``"KAB"`` ...).

    Args:
        threat_type: Raw label or missing value (``None``/``NaN``).

    Returns:
        A canonical threat key, or ``"unknown"`` when it cannot be resolved.

    Examples:
        >>> normalize_threat_type("Ballistic")
        'ballistic_missile'
        >>> normalize_threat_type("UAV wave")
        'uav_shahed'
        >>> normalize_threat_type(None)
        'unknown'
    """
    if threat_type is None or (isinstance(threat_type, float) and np.isnan(threat_type)):
        return "unknown"
    text = str(threat_type).strip().lower().replace(" ", "_").replace("-", "_")
    if not text or text in {"nan", "none", "null", "<na>"}:
        return "unknown"
    if text in THREAT_TYPES:
        return text
    if text in _THREAT_ALIASES:
        return _THREAT_ALIASES[text]
    for canonical in THREAT_TYPES:
        if canonical in text:
            return canonical
    for alias, canonical in _THREAT_ALIASES.items():
        if alias in text:
            return canonical
    return "unknown"


def classify_threat_wave(threat_type: object) -> str:
    """Map a free-form threat label to a kinematic wave class.

    Args:
        threat_type: Raw label (see :func:`normalize_threat_type`).

    Returns:
        ``"fast"``, ``"slow"`` or ``"unknown"``.

    Examples:
        >>> classify_threat_wave("Ballistic")
        'fast'
        >>> classify_threat_wave("uav_shahed")
        'slow'
        >>> classify_threat_wave(None)
        'unknown'
    """
    return THREAT_TYPES.get(normalize_threat_type(threat_type), ThreatWave.UNKNOWN).value


def to_utc_timestamp(value: TimeLike, *, name: str = "cutoff_time") -> pd.Timestamp:
    """Coerce a datetime-like value to a timezone-aware UTC ``Timestamp``.

    Naive inputs are *assumed* to be UTC. Aware inputs are converted.

    Args:
        value: ``datetime``, ``pd.Timestamp`` or ISO-8601 string.
        name: Parameter name used in error messages.

    Returns:
        A UTC-localised :class:`pandas.Timestamp`.

    Raises:
        TypeError: If ``value`` is not datetime-like.
        ValueError: If ``value`` cannot be parsed or is ``NaT``.
    """
    if isinstance(value, bool) or not isinstance(value, (datetime, pd.Timestamp, str)):
        raise TypeError(
            f"{name} must be a datetime, Timestamp or ISO string, got {type(value).__name__}"
        )
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} could not be parsed as a datetime: {value!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"{name} must not be NaT/None")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _as_utc(series: pd.Series) -> pd.Series:
    """Return a datetime series as tz-aware UTC at nanosecond resolution.

    Naive values are assumed to be UTC. The resolution is pinned to ``ns`` so
    CSV-parsed data (which pandas >= 3 parses at ``us``) and synthetic data
    share one exact dtype, ``datetime64[ns, UTC]``.
    """
    out = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
    return out.dt.as_unit("ns")


def _coerce_datetime_column(series: pd.Series, *, column: str) -> pd.Series:
    """Parse a column to tz-aware UTC datetimes, coercing bad values to NaT.

    Mixed naive/aware inputs are handled by parsing in UTC mode; naive values
    are therefore interpreted as UTC. A warning is logged when parsing drops
    values that were non-empty in the source.

    Args:
        series: Source column (strings, objects or datetimes).
        column: Column name for log messages.

    Returns:
        A ``datetime64[ns, UTC]`` series aligned to ``series.index``.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return _as_utc(series)

    raw = series.astype("string").str.strip()
    parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed").dt.as_unit("ns")
    non_empty = raw.notna() & (raw != "")
    bad = int((non_empty & parsed.isna()).sum())
    if bad:
        logger.warning(
            "Column %r: %d non-empty value(s) could not be parsed as datetimes and were set to NaT",
            column,
            bad,
        )
    return parsed


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def generate_synthetic_alarms(
    num_records: int = 2000,
    start_date: str = "2026-01-01",
    *,
    seed: Optional[int] = 42,
    mean_alarms_per_day: float = 60.0,
) -> pd.DataFrame:
    """Generate a realistic synthetic air-raid alarm log.

    The generator models three structural features of real Ukrainian alarm
    data so that downstream cascade logic has something meaningful to learn:

    * **Spatial skew**: front-line and hub oblasts (Kharkiv, Dnipropetrovsk,
      Zaporizhzhia, Odesa, Kyiv) are alarmed far more often than the west.
    * **Wave structure**: alarms arrive in bursts. Each burst is assigned one
      threat type; slow-wave (UAV) bursts fan out across several regions with
      staggered starts over ~1-3 hours, while fast-wave (ballistic / aviation)
      bursts hit regions nearly simultaneously.
    * **Diurnal pattern**: slow-wave bursts cluster at night (22:00-06:00
      local, approximated as 20:00-04:00 UTC); fast-wave bursts are spread
      more evenly, with aviation nudged towards daytime.

    Args:
        num_records: Number of alarm rows to produce. Must be ``>= 1``.
        start_date: ISO date (``YYYY-MM-DD``) of the first possible alarm.
            Interpreted as UTC midnight.
        seed: Seed for the NumPy random generator. ``None`` yields
            non-reproducible output.
        mean_alarms_per_day: Controls the temporal density of the log.

    Returns:
        A DataFrame in the canonical schema plus a ``source`` column set to
        ``"synthetic"``, sorted by ``start_time`` and containing exactly
        ``num_records`` rows.

    Raises:
        ValueError: If ``num_records < 1``, ``mean_alarms_per_day <= 0`` or
            ``start_date`` is not a parseable date.
    """
    if (
        isinstance(num_records, bool)
        or not isinstance(num_records, (int, np.integer))
        or num_records < 1
    ):
        raise ValueError(f"num_records must be a positive integer, got {num_records!r}")
    if mean_alarms_per_day <= 0:
        raise ValueError(f"mean_alarms_per_day must be > 0, got {mean_alarms_per_day!r}")
    origin = to_utc_timestamp(start_date, name="start_date").normalize()

    rng = np.random.default_rng(seed)
    regions = np.asarray(UKRAINIAN_REGIONS, dtype=object)
    region_p = np.asarray([_REGION_WEIGHTS[r] for r in UKRAINIAN_REGIONS], dtype=float)
    region_p /= region_p.sum()
    threat_keys = list(_THREAT_WEIGHTS)
    threat_p = np.asarray([_THREAT_WEIGHTS[k] for k in threat_keys], dtype=float)
    threat_p /= threat_p.sum()

    # Average burst size is ~4-5 regions => bursts/day ~= alarms/day / 4.5.
    burst_rate_per_min = (mean_alarms_per_day / 4.5) / (24 * 60)

    rows_region: list[str] = []
    rows_start: list[pd.Timestamp] = []
    rows_threat: list[str] = []
    clock = origin

    while len(rows_region) < num_records:
        clock = clock + pd.Timedelta(minutes=float(rng.exponential(1.0 / burst_rate_per_min)))
        threat = str(rng.choice(threat_keys, p=threat_p))
        wave = THREAT_TYPES[threat]

        # Nudge the burst into a realistic hour-of-day band.
        hour_utc = clock.hour
        if wave is ThreatWave.SLOW and 4 <= hour_utc < 20 and rng.random() < 0.7:
            # Push most UAV waves to the following night.
            clock = clock + pd.Timedelta(hours=(20 - hour_utc) + float(rng.uniform(0, 6)))
        elif threat == "aviation" and (hour_utc < 5 or hour_utc >= 21) and rng.random() < 0.5:
            clock = clock + pd.Timedelta(hours=float(rng.uniform(4, 10)))

        if wave is ThreatWave.SLOW:
            burst_size = int(np.clip(rng.poisson(6.0), 1, 14))
            # UAVs crawl across oblasts: cumulative exponential gaps (~25 min).
            offsets = np.cumsum(rng.exponential(25.0, size=burst_size))
        else:
            burst_size = int(np.clip(rng.poisson(3.0), 1, 10))
            # Ballistic / aviation alarms are near-simultaneous.
            offsets = np.sort(np.abs(rng.normal(0.0, 1.5, size=burst_size)))

        chosen = rng.choice(regions, size=burst_size, replace=False, p=region_p)
        for region, offset in zip(chosen, offsets):
            rows_region.append(str(region))
            rows_start.append(clock + pd.Timedelta(minutes=float(offset)))
            rows_threat.append(threat)

    # The last burst may overshoot; truncate to the exact requested size.
    rows_region = rows_region[:num_records]
    rows_start = rows_start[:num_records]
    rows_threat = rows_threat[:num_records]

    start = pd.DatetimeIndex(rows_start).floor("s")
    medians = np.asarray([_DURATION_PARAMS[t][0] for t in rows_threat])
    sigmas = np.asarray([_DURATION_PARAMS[t][1] for t in rows_threat])
    durations = np.clip(rng.lognormal(mean=np.log(medians), sigma=sigmas), 3.0, 600.0)
    end = start + pd.to_timedelta(np.round(durations), unit="m")

    df = pd.DataFrame(
        {
            "region": rows_region,
            "start_time": start,
            "end_time": end,
            "threat_type": rows_threat,
        }
    )
    # ~1.5% of real records have no recorded end (feed outages) -> NaT.
    missing_end = rng.random(len(df)) < 0.015
    df.loc[missing_end, "end_time"] = pd.NaT

    df = _finalize(df)
    df["source"] = "synthetic"
    logger.info(
        "Generated %d synthetic alarms spanning %s -> %s",
        len(df),
        df["start_time"].iloc[0],
        df["start_time"].iloc[-1],
    )
    return df


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
@dataclass
class AirThreatLoader:
    """Load, clean and temporally safeguard air-raid alarm logs.

    The loader holds configuration only; every method takes inputs and
    returns a new DataFrame, so a single instance can be shared across
    backtests that use different cutoffs.

    Attributes:
        column_aliases: Lower-cased source header -> canonical column mapping
            used to normalise heterogeneous CSV exports. Defaults to
            :data:`DEFAULT_COLUMN_ALIASES`; pass a custom mapping to extend it.
        mask_future_end_times: When ``True`` (default), any ``end_time`` that
            is ``>= cutoff_time`` is replaced with ``NaT`` and the duration
            recomputed, because that end would not be known at ``T``.
        drop_unparseable_start: When ``True`` (default), rows whose
            ``start_time`` cannot be parsed are dropped; otherwise a
            :class:`SchemaValidationError` is raised.
        min_duration_min: Rows with ``end_time < start_time`` always get their
            ``end_time`` masked; rows with a duration below this threshold
            (minutes) are masked as well rather than dropped.
        csv_read_kwargs: Extra keyword arguments forwarded to
            :func:`pandas.read_csv` (encoding, separator, ...).
    """

    column_aliases: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_COLUMN_ALIASES)
    )
    mask_future_end_times: bool = True
    drop_unparseable_start: bool = True
    min_duration_min: float = 0.0
    csv_read_kwargs: Mapping[str, object] = field(default_factory=dict)

    # ---- public API ----------------------------------------------------- #
    def load_csv(self, csv_path: PathLike) -> pd.DataFrame:
        """Read a raw alarm CSV without any cleaning.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            The raw DataFrame exactly as read by pandas.

        Raises:
            FileNotFoundError: If the path does not exist or is not a file.
            SchemaValidationError: If the file is empty or unreadable as CSV.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Alarm CSV not found: {path}")
        logger.info("Reading alarm log from %s", path)
        try:
            raw = pd.read_csv(path, **dict(self.csv_read_kwargs))
        except pd.errors.EmptyDataError as exc:
            raise SchemaValidationError(f"CSV is empty: {path}") from exc
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise SchemaValidationError(f"CSV could not be parsed: {path} ({exc})") from exc
        logger.debug("Read %d raw rows, columns=%s", len(raw), list(raw.columns))
        return raw

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise a raw alarm frame into the canonical schema.

        Steps, in order:

        1. Resolve column aliases (case/whitespace-insensitive).
        2. Validate presence of :data:`REQUIRED_SOURCE_COLUMNS`.
        3. Parse ``start_time`` / ``end_time`` to UTC datetimes (bad -> NaT).
        4. Drop rows with missing region or unparseable start time.
        5. Normalise region strings and threat types; derive ``wave_class``.
        6. Mask impossible end times (before start) and compute duration.
        7. Deduplicate on (region, start_time, threat_type), sort, re-index.

        Args:
            raw: DataFrame as read from disk or produced upstream.

        Returns:
            A new DataFrame with columns :data:`CANONICAL_COLUMNS` first,
            followed by any extra source columns.

        Raises:
            TypeError: If ``raw`` is not a DataFrame.
            SchemaValidationError: If required columns are missing, or if
                ``drop_unparseable_start`` is ``False`` and bad starts exist.
        """
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f"clean() expects a DataFrame, got {type(raw).__name__}")
        df = self._resolve_columns(raw)

        missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"Missing required column(s) {missing}; available: {list(raw.columns)}"
            )

        n_in = len(df)
        df["start_time"] = _coerce_datetime_column(df["start_time"], column="start_time")
        if "end_time" in df.columns:
            df["end_time"] = _coerce_datetime_column(df["end_time"], column="end_time")
        else:
            df["end_time"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

        bad_start = df["start_time"].isna()
        if bad_start.any():
            if not self.drop_unparseable_start:
                raise SchemaValidationError(
                    f"{int(bad_start.sum())} row(s) have missing/unparseable start_time"
                )
            logger.warning(
                "Dropping %d row(s) with missing/unparseable start_time", int(bad_start.sum())
            )
            df = df.loc[~bad_start]

        region = df["region"].astype("string").str.strip()
        bad_region = region.isna() | (region == "")
        if bad_region.any():
            logger.warning("Dropping %d row(s) with missing region", int(bad_region.sum()))
            df = df.loc[~bad_region]
            region = region.loc[~bad_region]
        df["region"] = region.astype(str)

        if "threat_type" not in df.columns:
            df["threat_type"] = "unknown"

        df = _finalize(df, min_duration_min=self.min_duration_min)
        logger.info("Cleaned alarm log: %d -> %d rows", n_in, len(df))
        return df

    def apply_temporal_cutoff(self, df: pd.DataFrame, cutoff_time: TimeLike) -> pd.DataFrame:
        """Drop every record that would not be observable strictly before ``T``.

        Rule: a row survives **iff** ``start_time < cutoff_time``. Rows with
        ``start_time >= cutoff_time`` are removed. If
        :attr:`mask_future_end_times` is set, surviving rows whose
        ``end_time >= cutoff_time`` have ``end_time`` set to ``NaT`` and
        ``duration_min`` to ``NaN`` (the alarm is still active at ``T``).

        Args:
            df: A cleaned DataFrame (must contain a datetime ``start_time``).
            cutoff_time: The observation instant ``T``. Naive values are
                treated as UTC.

        Returns:
            A new, chronologically sorted DataFrame with a fresh index.

        Raises:
            SchemaValidationError: If ``start_time`` is missing or not a
                datetime column.
            TemporalLeakageError: If post-filter verification fails.
        """
        if "start_time" not in df.columns or not pd.api.types.is_datetime64_any_dtype(
            df["start_time"]
        ):
            raise SchemaValidationError(
                "apply_temporal_cutoff requires a datetime 'start_time' column; call clean() first"
            )
        cutoff = to_utc_timestamp(cutoff_time)

        start = _as_utc(df["start_time"])
        keep = start < cutoff
        n_dropped = int((~keep).sum())
        out = df.loc[keep].copy()
        out["start_time"] = start.loc[keep]

        if (
            self.mask_future_end_times
            and "end_time" in out.columns
            and pd.api.types.is_datetime64_any_dtype(out["end_time"])
        ):
            end = _as_utc(out["end_time"])
            future_end = end.notna() & (end >= cutoff)
            if future_end.any():
                logger.info(
                    "Masking %d end_time value(s) at/after cutoff (alarm still active at T)",
                    int(future_end.sum()),
                )
                end = end.mask(future_end, pd.NaT)
            out["end_time"] = end
            out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0

        out = out.sort_values(["start_time", "region"], kind="mergesort").reset_index(drop=True)
        self.verify_no_leakage(out, cutoff)
        logger.info(
            "Temporal cutoff %s: kept %d rows, dropped %d (start_time >= T)",
            cutoff.isoformat(),
            len(out),
            n_dropped,
        )
        return out

    def load_and_filter_alarms(self, csv_path: PathLike, cutoff_time: TimeLike) -> pd.DataFrame:
        """Read, clean and temporally filter an alarm CSV in one call.

        Args:
            csv_path: Path to the CSV file.
            cutoff_time: Observation instant ``T``; rows with
                ``start_time >= T`` are dropped.

        Returns:
            A clean, sorted DataFrame containing only alarms that started
            strictly before ``cutoff_time``.

        Raises:
            FileNotFoundError: If ``csv_path`` does not exist.
            SchemaValidationError: On unusable files or schemas.
            TemporalLeakageError: If the leakage tripwire fires.
        """
        raw = self.load_csv(csv_path)
        cleaned = self.clean(raw)
        return self.apply_temporal_cutoff(cleaned, cutoff_time)

    def load_or_generate(
        self,
        csv_path: Optional[PathLike],
        cutoff_time: TimeLike,
        *,
        num_records: int = 2000,
        start_date: str = "2026-01-01",
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Load from CSV when available, else fall back to synthetic data.

        The fallback triggers when ``csv_path`` is ``None`` or the file does
        not exist. Schema or parse errors in an *existing* file are **not**
        swallowed: silently replacing a corrupt real dataset with synthetic
        data would be far more dangerous than failing.

        The synthetic data is subject to the same temporal cutoff.

        Args:
            csv_path: Optional path to a CSV file.
            cutoff_time: Observation instant ``T``.
            num_records: Rows to synthesise on fallback.
            start_date: First synthetic alarm date on fallback.
            seed: Random seed on fallback.

        Returns:
            A cutoff-filtered DataFrame from either source. Real data rows
            carry ``source == "csv"`` (unless the file already had a
            ``source`` column); synthetic rows carry ``source == "synthetic"``.
        """
        if csv_path is not None and Path(csv_path).is_file():
            df = self.load_and_filter_alarms(csv_path, cutoff_time)
            if "source" not in df.columns:
                df["source"] = "csv"
            return df

        logger.warning(
            "Alarm CSV %r not available; falling back to %d synthetic records from %s",
            None if csv_path is None else str(csv_path),
            num_records,
            start_date,
        )
        synthetic = generate_synthetic_alarms(
            num_records=num_records, start_date=start_date, seed=seed
        )
        return self.apply_temporal_cutoff(synthetic, cutoff_time)

    @staticmethod
    def verify_no_leakage(df: pd.DataFrame, cutoff_time: TimeLike) -> None:
        """Assert that no row in ``df`` starts at or after ``cutoff_time``.

        Args:
            df: Any DataFrame with a datetime ``start_time`` column.
            cutoff_time: Observation instant ``T``.

        Raises:
            TemporalLeakageError: If any ``start_time >= T`` is found.
        """
        if df.empty:
            return
        cutoff = to_utc_timestamp(cutoff_time)
        leaked = int((_as_utc(df["start_time"]) >= cutoff).sum())
        if leaked:
            raise TemporalLeakageError(
                f"{leaked} row(s) have start_time >= cutoff {cutoff.isoformat()}; "
                "temporal filter failed"
            )

    # ---- internals ------------------------------------------------------ #
    def _resolve_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rename source columns to canonical names using the alias map.

        The first source column matching a canonical name wins; later
        duplicates are left untouched (and logged) so no data is lost.

        Args:
            raw: Source DataFrame.

        Returns:
            A renamed copy of ``raw``.
        """
        aliases = {k.strip().lower(): v for k, v in self.column_aliases.items()}
        rename: dict[str, str] = {}
        taken: set[str] = set()
        for col in raw.columns:
            canonical = aliases.get(str(col).strip().lower())
            if canonical is None:
                continue
            if canonical in taken:
                logger.debug(
                    "Column %r also maps to %r but that target is taken; leaving as-is",
                    col,
                    canonical,
                )
                continue
            rename[col] = canonical
            taken.add(canonical)
        if rename:
            logger.debug("Resolved column aliases: %s", rename)
        return raw.rename(columns=rename).copy()


# --------------------------------------------------------------------------- #
# Shared finalisation
# --------------------------------------------------------------------------- #
def _finalize(df: pd.DataFrame, *, min_duration_min: float = 0.0) -> pd.DataFrame:
    """Derive wave class and duration, deduplicate, sort and order columns.

    Shared by :meth:`AirThreatLoader.clean` and
    :func:`generate_synthetic_alarms` so both paths emit an identical schema.

    Args:
        df: Frame with ``region``, ``start_time``, ``end_time`` (UTC datetimes)
            and ``threat_type`` columns.
        min_duration_min: Durations below this (minutes) get ``end_time``
            masked to ``NaT``.

    Returns:
        A new DataFrame ordered as :data:`CANONICAL_COLUMNS` + extras.
    """
    out = df.copy()

    out["threat_type"] = out["threat_type"].map(normalize_threat_type).astype(str)
    out["wave_class"] = out["threat_type"].map(classify_threat_wave).astype(str)

    # Impossible / degenerate end times are masked, not dropped: the alarm
    # start is still a valid observation even when the end is garbage.
    invalid_end = out["end_time"].notna() & (out["end_time"] < out["start_time"])
    if invalid_end.any():
        logger.warning(
            "Masking %d end_time value(s) earlier than start_time", int(invalid_end.sum())
        )
        out["end_time"] = out["end_time"].mask(invalid_end, pd.NaT)

    out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
    if min_duration_min > 0:
        too_short = out["duration_min"].notna() & (out["duration_min"] < min_duration_min)
        if too_short.any():
            logger.warning(
                "Masking %d end_time value(s) with duration < %.1f min",
                int(too_short.sum()),
                min_duration_min,
            )
            out["end_time"] = out["end_time"].mask(too_short, pd.NaT)
            out["duration_min"] = out["duration_min"].mask(too_short, np.nan)

    before = len(out)
    out = out.drop_duplicates(subset=["region", "start_time", "threat_type"], keep="first")
    if len(out) != before:
        logger.info("Dropped %d duplicate alarm row(s)", before - len(out))

    out = out.sort_values(["start_time", "region"], kind="mergesort").reset_index(drop=True)

    if "alarm_id" in out.columns:
        ids = pd.to_numeric(out["alarm_id"], errors="coerce")
        if ids.isna().any() or ids.duplicated().any():
            logger.debug("Source alarm_id unusable (missing or duplicated); regenerating ids")
            out["alarm_id"] = np.arange(len(out), dtype=np.int64)
        else:
            out["alarm_id"] = ids.astype(np.int64)
    else:
        out["alarm_id"] = np.arange(len(out), dtype=np.int64)

    extras = [c for c in out.columns if c not in CANONICAL_COLUMNS]
    return out[list(CANONICAL_COLUMNS) + extras]


# --------------------------------------------------------------------------- #
# Module-level convenience function (spec-mandated signature)
# --------------------------------------------------------------------------- #
def load_and_filter_alarms(csv_path: str, cutoff_time: datetime) -> pd.DataFrame:
    """Load an alarm CSV and drop every row with ``start_time >= cutoff_time``.

    Thin wrapper around :meth:`AirThreatLoader.load_and_filter_alarms` using
    default loader settings. See that method for details and raised errors.

    Args:
        csv_path: Path to the CSV file.
        cutoff_time: Observation instant ``T``. Naive datetimes are treated as
            UTC.

    Returns:
        A clean, chronologically sorted DataFrame with only pre-cutoff alarms.
    """
    return AirThreatLoader().load_and_filter_alarms(csv_path, cutoff_time)


# --------------------------------------------------------------------------- #
# Self-check entry point:  python -m src.loader
# --------------------------------------------------------------------------- #
def _self_check() -> int:
    """Run a quick smoke test on synthetic data and print a summary.

    Returns:
        Process exit code (``0`` on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = generate_synthetic_alarms(num_records=2000, start_date="2026-01-01", seed=7)
    cutoff = df["start_time"].iloc[len(df) // 2]
    filtered = AirThreatLoader().apply_temporal_cutoff(df, cutoff)
    assert (filtered["start_time"] < cutoff).all(), "start_time leakage"
    assert filtered["end_time"].dropna().lt(cutoff).all(), "end_time leakage"

    print(df.head(8).to_string())
    print("\nwave_class mix:")
    print(df["wave_class"].value_counts(normalize=True).round(3).to_string())
    print("\nmedian duration by wave (min):")
    print(df.groupby("wave_class")["duration_min"].median().round(1).to_string())
    print(f"\ncutoff={cutoff.isoformat()}  rows before={len(df)}  after={len(filtered)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_check())
