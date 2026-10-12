"""Tests for ``src.loader``.

Run with ``python -m unittest discover tests`` (no extra dependencies) or
``python -m pytest tests/`` if pytest is installed.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.loader import (
    CANONICAL_COLUMNS,
    UKRAINIAN_REGIONS,
    AirThreatLoader,
    SchemaValidationError,
    TemporalLeakageError,
    classify_threat_wave,
    generate_synthetic_alarms,
    load_and_filter_alarms,
    normalize_threat_type,
    to_utc_timestamp,
)

UTC = timezone.utc
FAR_FUTURE = datetime(2100, 1, 1, tzinfo=UTC)


def _write_csv(tmp: Path, text: str, name: str = "alarms.csv") -> Path:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return path


class SyntheticDataTests(unittest.TestCase):
    def test_shape_schema_and_order(self) -> None:
        df = generate_synthetic_alarms(num_records=500, start_date="2026-02-01", seed=1)
        self.assertEqual(len(df), 500)
        self.assertEqual(list(df.columns[: len(CANONICAL_COLUMNS)]), list(CANONICAL_COLUMNS))
        self.assertTrue(df["start_time"].is_monotonic_increasing)
        self.assertEqual(str(df["start_time"].dt.tz), "UTC")
        self.assertGreaterEqual(df["start_time"].min(), pd.Timestamp("2026-02-01", tz="UTC"))
        self.assertTrue(set(df["region"]).issubset(UKRAINIAN_REGIONS))
        self.assertTrue(set(df["wave_class"]).issubset({"fast", "slow"}))
        self.assertTrue((df["source"] == "synthetic").all())
        self.assertTrue(df["alarm_id"].is_unique)

    def test_reproducible_with_seed(self) -> None:
        a = generate_synthetic_alarms(300, seed=123)
        b = generate_synthetic_alarms(300, seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_wave_classes_have_distinct_durations(self) -> None:
        df = generate_synthetic_alarms(2000, seed=5)
        med = df.groupby("wave_class")["duration_min"].median()
        self.assertGreater(med["slow"], 2 * med["fast"])
        self.assertTrue((df["duration_min"].dropna() >= 3).all())
        share = df["wave_class"].value_counts(normalize=True)
        self.assertGreater(share["slow"], 0.3)
        self.assertGreater(share["fast"], 0.3)

    def test_end_after_start_or_nat(self) -> None:
        df = generate_synthetic_alarms(1000, seed=9)
        valid = df["end_time"].notna()
        self.assertTrue((df.loc[valid, "end_time"] > df.loc[valid, "start_time"]).all())
        self.assertTrue(df["duration_min"].isna().any())  # some feed-outage NaTs

    def test_invalid_args(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_alarms(0)
        with self.assertRaises(ValueError):
            generate_synthetic_alarms(10, start_date="not-a-date")
        with self.assertRaises(ValueError):
            generate_synthetic_alarms(10, mean_alarms_per_day=0)

    def test_single_record(self) -> None:
        self.assertEqual(len(generate_synthetic_alarms(1, seed=0)), 1)


class ClassificationTests(unittest.TestCase):
    def test_fast_and_slow(self) -> None:
        for label in ("ballistic_missile", "Ballistic", "ISKANDER", "aviation", "cruise missile", "KAB"):
            self.assertEqual(classify_threat_wave(label), "fast", label)
        for label in ("uav_shahed", "Shahed", "drones", "UAV wave", "uav_recon"):
            self.assertEqual(classify_threat_wave(label), "slow", label)

    def test_unknown(self) -> None:
        for label in (None, float("nan"), "", "   ", "meteor", "null"):
            self.assertEqual(classify_threat_wave(label), "unknown")
            self.assertEqual(normalize_threat_type(label), "unknown")


class TemporalCutoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = AirThreatLoader()
        self.df = generate_synthetic_alarms(1000, seed=3)

    def test_strict_inequality_at_boundary(self) -> None:
        boundary = self.df["start_time"].iloc[400]
        out = self.loader.apply_temporal_cutoff(self.df, boundary)
        self.assertTrue((out["start_time"] < boundary).all())
        self.assertFalse((out["start_time"] == boundary).any())  # == T is dropped
        self.assertEqual(len(out), int((self.df["start_time"] < boundary).sum()))

    def test_end_times_masked_at_cutoff(self) -> None:
        cutoff = self.df["start_time"].iloc[500]
        out = self.loader.apply_temporal_cutoff(self.df, cutoff)
        self.assertTrue(out["end_time"].dropna().lt(cutoff).all())
        masked = out["end_time"].isna()
        self.assertTrue(masked.any(), "expected some still-active alarms to be masked")
        self.assertTrue(out.loc[masked, "duration_min"].isna().all())

    def test_end_times_kept_when_masking_disabled(self) -> None:
        loader = AirThreatLoader(mask_future_end_times=False)
        cutoff = self.df["start_time"].iloc[500]
        out = loader.apply_temporal_cutoff(self.df, cutoff)
        self.assertTrue((out["start_time"] < cutoff).all())
        self.assertTrue(out["end_time"].dropna().ge(cutoff).any())

    def test_naive_cutoff_treated_as_utc(self) -> None:
        aware = self.df["start_time"].iloc[300].to_pydatetime()
        naive = aware.replace(tzinfo=None)
        a = self.loader.apply_temporal_cutoff(self.df, aware)
        b = self.loader.apply_temporal_cutoff(self.df, naive)
        pd.testing.assert_frame_equal(a, b)

    def test_non_utc_cutoff_converted(self) -> None:
        ts = self.df["start_time"].iloc[300]
        a = self.loader.apply_temporal_cutoff(self.df, ts)
        b = self.loader.apply_temporal_cutoff(self.df, ts.tz_convert("Europe/Kyiv"))
        pd.testing.assert_frame_equal(a, b)

    def test_cutoff_before_all_data_gives_empty(self) -> None:
        out = self.loader.apply_temporal_cutoff(self.df, datetime(2000, 1, 1, tzinfo=UTC))
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), list(self.df.columns))

    def test_cutoff_after_all_data_keeps_everything(self) -> None:
        out = self.loader.apply_temporal_cutoff(self.df, FAR_FUTURE)
        self.assertEqual(len(out), len(self.df))

    def test_bad_cutoff_types(self) -> None:
        with self.assertRaises(TypeError):
            self.loader.apply_temporal_cutoff(self.df, 12345)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.loader.apply_temporal_cutoff(self.df, "garbage")

    def test_requires_cleaned_frame(self) -> None:
        with self.assertRaises(SchemaValidationError):
            self.loader.apply_temporal_cutoff(
                pd.DataFrame({"start_time": ["2026-01-01"]}), "2026-02-01"
            )

    def test_verify_no_leakage_tripwire(self) -> None:
        with self.assertRaises(TemporalLeakageError):
            AirThreatLoader.verify_no_leakage(self.df, self.df["start_time"].iloc[10])
        AirThreatLoader.verify_no_leakage(self.df, FAR_FUTURE)  # must not raise

    def test_to_utc_timestamp(self) -> None:
        self.assertEqual(str(to_utc_timestamp("2026-01-01").tz), "UTC")
        self.assertEqual(
            to_utc_timestamp(datetime(2026, 1, 1, 12, tzinfo=UTC)),
            to_utc_timestamp("2026-01-01 12:00"),
        )
        with self.assertRaises(ValueError):
            to_utc_timestamp("NaT")


class CsvLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.loader = AirThreatLoader()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_aliases_cleaning_and_sorting(self) -> None:
        csv = (
            "id,region_title,start,end,alert_type\n"
            "3,Kyiv City,2026-01-02 10:00:00,2026-01-02 10:30:00,Shahed\n"
            "1, Odesa Oblast ,2026-01-01T08:00:00Z,2026-01-01T08:20:00Z,ballistic\n"
            "2,Lviv Oblast,2026-01-01 23:00:00,,\n"
            "4,Sumy Oblast,not a date,2026-01-02 11:00:00,aviation\n"
            "5,,2026-01-03 00:00:00,2026-01-03 01:00:00,uav\n"
            "6,Kharkiv Oblast,2026-01-03 05:00:00,2026-01-03 04:00:00,KAB\n"
            "1,Odesa Oblast,2026-01-01 08:00:00,2026-01-01 08:20:00,ballistic\n"
        )
        path = _write_csv(self.tmp, csv)
        df = self.loader.load_and_filter_alarms(path, FAR_FUTURE)

        self.assertEqual(list(df.columns[: len(CANONICAL_COLUMNS)]), list(CANONICAL_COLUMNS))
        self.assertEqual(
            df["region"].tolist(), ["Odesa Oblast", "Lviv Oblast", "Kyiv City", "Kharkiv Oblast"]
        )
        self.assertTrue(df["start_time"].is_monotonic_increasing)
        self.assertEqual(
            df["threat_type"].tolist(),
            ["ballistic_missile", "unknown", "uav_shahed", "guided_bomb"],
        )
        self.assertEqual(df["wave_class"].tolist(), ["fast", "unknown", "slow", "fast"])
        # end before start -> masked; missing end -> NaT
        self.assertTrue(pd.isna(df.loc[df["region"] == "Kharkiv Oblast", "end_time"]).all())
        self.assertTrue(pd.isna(df.loc[df["region"] == "Lviv Oblast", "duration_min"]).all())
        self.assertAlmostEqual(
            float(df.loc[df["region"] == "Odesa Oblast", "duration_min"].iloc[0]), 20.0
        )
        self.assertEqual(str(df["start_time"].dt.tz), "UTC")
        self.assertTrue(df["alarm_id"].is_unique)

    def test_cutoff_applied_from_csv(self) -> None:
        csv = (
            "region,start_time,end_time,threat_type\n"
            "Kyiv City,2026-01-01 10:00,2026-01-01 12:00,uav\n"
            "Kyiv City,2026-01-01 11:00,2026-01-01 11:15,ballistic\n"
            "Kyiv City,2026-01-01 12:00,2026-01-01 12:30,aviation\n"
        )
        path = _write_csv(self.tmp, csv)
        cutoff = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
        df = load_and_filter_alarms(str(path), cutoff)
        self.assertEqual(len(df), 1)  # 11:00 (== T) and 12:00 dropped
        self.assertTrue(pd.isna(df["end_time"].iloc[0]))  # 12:00 end is unknown at T

    def test_missing_required_column(self) -> None:
        path = _write_csv(self.tmp, "region,threat\nKyiv City,uav\n")
        with self.assertRaises(SchemaValidationError):
            self.loader.load_and_filter_alarms(path, FAR_FUTURE)

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.loader.load_and_filter_alarms(self.tmp / "nope.csv", FAR_FUTURE)

    def test_empty_file(self) -> None:
        path = _write_csv(self.tmp, "")
        with self.assertRaises(SchemaValidationError):
            self.loader.load_csv(path)

    def test_strict_mode_raises_on_bad_start(self) -> None:
        path = _write_csv(self.tmp, "region,start_time\nKyiv City,garbage\n")
        strict = AirThreatLoader(drop_unparseable_start=False)
        with self.assertRaises(SchemaValidationError):
            strict.load_and_filter_alarms(path, FAR_FUTURE)

    def test_min_duration_masking(self) -> None:
        path = _write_csv(
            self.tmp, "region,start_time,end_time\nKyiv City,2026-01-01 10:00,2026-01-01 10:00:30\n"
        )
        df = AirThreatLoader(min_duration_min=1.0).load_and_filter_alarms(path, FAR_FUTURE)
        self.assertTrue(pd.isna(df["end_time"].iloc[0]))

    def test_load_or_generate_fallback_and_real(self) -> None:
        cutoff = datetime(2026, 1, 10, tzinfo=UTC)
        synthetic = self.loader.load_or_generate(
            self.tmp / "missing.csv", cutoff, num_records=400, seed=2
        )
        self.assertTrue((synthetic["source"] == "synthetic").all())
        self.assertTrue((synthetic["start_time"] < pd.Timestamp(cutoff)).all())
        self.assertGreater(len(synthetic), 0)

        path = _write_csv(self.tmp, "region,start_time\nKyiv City,2026-01-01 10:00\n")
        real = self.loader.load_or_generate(path, cutoff)
        self.assertEqual(real["source"].tolist(), ["csv"])

        none_path = self.loader.load_or_generate(None, cutoff, num_records=50, seed=0)
        self.assertTrue((none_path["source"] == "synthetic").all())

    def test_corrupt_existing_file_is_not_swallowed(self) -> None:
        path = _write_csv(self.tmp, "foo,bar\n1,2\n")
        with self.assertRaises(SchemaValidationError):
            self.loader.load_or_generate(path, FAR_FUTURE)

    def test_roundtrip_synthetic_through_csv(self) -> None:
        df = generate_synthetic_alarms(200, seed=11)
        path = self.tmp / "synthetic.csv"
        df.to_csv(path, index=False)
        cutoff = df["start_time"].iloc[120]
        reloaded = self.loader.load_and_filter_alarms(path, cutoff)
        direct = self.loader.apply_temporal_cutoff(df, cutoff)
        self.assertEqual(len(reloaded), len(direct))
        pd.testing.assert_series_equal(
            reloaded["start_time"], direct["start_time"], check_names=False
        )
        self.assertEqual(reloaded["wave_class"].tolist(), direct["wave_class"].tolist())
        self.assertEqual(reloaded["alarm_id"].tolist(), direct["alarm_id"].tolist())


if __name__ == "__main__":
    unittest.main()
