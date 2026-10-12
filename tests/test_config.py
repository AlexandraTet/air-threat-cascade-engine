"""Tests for ``src.config``. Run with ``python -m unittest discover tests``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import (
    DEFAULTS,
    Settings,
    get_settings,
    load_dotenv_file,
    load_settings,
    parse_dotenv,
    reset_settings_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ParseDotenvTests(unittest.TestCase):
    def test_parses_common_forms(self) -> None:
        text = """
        # comment
        PLAIN=value
        export EXPORTED=1
        QUOTED="hello world"  # inline comment after a quoted value is dropped
        SINGLE='single # not a comment'
        ESCAPED="line1\\nline2"
        INLINE=abc # comment
        EMPTY=""
        URL=https://example.com/path?a=1#frag
        """
        parsed = parse_dotenv(text)
        self.assertEqual(parsed["PLAIN"], "value")
        self.assertEqual(parsed["EXPORTED"], "1")
        self.assertEqual(parsed["QUOTED"], "hello world")
        self.assertEqual(parsed["SINGLE"], "single # not a comment")
        self.assertEqual(parsed["ESCAPED"], "line1\nline2")
        self.assertEqual(parsed["INLINE"], "abc")
        self.assertEqual(parsed["EMPTY"], "")
        self.assertEqual(parsed["URL"], "https://example.com/path?a=1#frag")

    def test_skips_malformed_lines(self) -> None:
        with self.assertLogs("src.config", level="WARNING"):
            parsed = parse_dotenv("NOEQUALS\n1BAD=x\n-also-bad=y\nGOOD=1\n")
        self.assertEqual(parsed, {"GOOD": "1"})


class LoadDotenvFileTests(unittest.TestCase):
    def test_missing_file_is_fine(self) -> None:
        env: dict[str, str] = {}
        self.assertEqual(load_dotenv_file("definitely/not/here.env", environ=env), {})
        self.assertEqual(env, {})

    def test_existing_vars_win_unless_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('GEMINI_API_KEY="from-file"\nLOG_LEVEL=DEBUG\n', encoding="utf-8")
            env = {"GEMINI_API_KEY": "from-process"}
            applied = load_dotenv_file(path, environ=env)
            self.assertEqual(applied, {"LOG_LEVEL": "DEBUG"})
            self.assertEqual(env["GEMINI_API_KEY"], "from-process")
            load_dotenv_file(path, environ=env, override=True)
            self.assertEqual(env["GEMINI_API_KEY"], "from-file")


class LoadSettingsTests(unittest.TestCase):
    def test_defaults_without_key_enable_offline_mode(self) -> None:
        settings = load_settings(env_file=None, environ={})
        self.assertIsNone(settings.gemini_api_key)
        self.assertFalse(settings.has_gemini_key)
        self.assertFalse(settings.llm_enabled)
        self.assertTrue(settings.offline_mode)
        self.assertTrue(settings.synthetic_fallback)
        self.assertEqual(settings.synthetic_records, 2000)
        self.assertEqual(settings.default_cutoff_timezone, "UTC")
        self.assertEqual(settings.masked_key(), "<not set>")
        self.assertTrue(any("offline mode" in w for w in settings.warnings))
        self.assertIn("offline", settings.describe())

    def test_key_enables_llm_and_is_masked(self) -> None:
        settings = load_settings(env_file=None, environ={"GEMINI_API_KEY": "AIzaSyExampleKey1234"})
        self.assertTrue(settings.llm_enabled)
        self.assertFalse(settings.offline_mode)
        self.assertEqual(settings.masked_key(), "AIza…1234")
        self.assertNotIn("AIzaSyExampleKey1234", settings.describe())
        self.assertEqual(settings.as_dict()["gemini_api_key"], "AIza…1234")
        self.assertFalse(any("offline" in w for w in settings.warnings))

    def test_blank_key_counts_as_missing(self) -> None:
        settings = load_settings(env_file=None, environ={"GEMINI_API_KEY": "   "})
        self.assertFalse(settings.has_gemini_key)

    def test_llm_hard_switch(self) -> None:
        settings = load_settings(env_file=None, environ={"GEMINI_API_KEY": "k" * 20, "LLM_ENABLED": "off"})
        self.assertTrue(settings.has_gemini_key)
        self.assertFalse(settings.llm_enabled)
        self.assertTrue(any("LLM_ENABLED is false" in w for w in settings.warnings))

    def test_invalid_values_fall_back_without_crashing(self) -> None:
        bad = {
            "SYNTHETIC_RECORDS": "lots",
            "SYNTHETIC_SEED": "-5",
            "SYNTHETIC_FALLBACK": "maybe",
            "SYNTHETIC_START_DATE": "01/02/2026",
            "DEFAULT_CUTOFF_TIMEZONE": "Mars/Olympus",
            "ACTIVE_WINDOW_HOURS": "soon",
            "CASCADE_MAX_LAG_MIN": "0",
            "LOG_LEVEL": "LOUD",
        }
        with self.assertLogs("src.config", level="WARNING"):
            settings = load_settings(env_file=None, environ=bad)
        self.assertEqual(settings.synthetic_records, 2000)
        self.assertEqual(settings.synthetic_seed, 42)
        self.assertTrue(settings.synthetic_fallback)
        self.assertEqual(settings.synthetic_start_date, "2026-01-01")
        self.assertEqual(settings.default_cutoff_timezone, "UTC")
        self.assertEqual(settings.active_window_hours, 8.0)
        self.assertEqual(settings.cascade_max_lag_min, 90.0)
        self.assertEqual(settings.log_level, "INFO")
        self.assertGreaterEqual(len(settings.warnings), len(bad))

    def test_valid_overrides_are_applied(self) -> None:
        good = {
            "SYNTHETIC_RECORDS": "500",
            "SYNTHETIC_FALLBACK": "no",
            "DEFAULT_CUTOFF_TIMEZONE": "Europe/Kyiv",
            "APP_ENV": "Production",
            "LOG_LEVEL": "debug",
            "ALARMS_CSV_PATH": "nope/alarms.csv",
        }
        settings = load_settings(env_file=None, environ=good)
        self.assertEqual(settings.synthetic_records, 500)
        self.assertFalse(settings.synthetic_fallback)
        self.assertEqual(settings.default_cutoff_timezone, "Europe/Kyiv")
        self.assertEqual(str(settings.cutoff_tzinfo), "Europe/Kyiv")
        self.assertEqual(settings.app_env, "production")
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertFalse(settings.csv_available)
        self.assertTrue(any("SYNTHETIC_FALLBACK is false" in w for w in settings.warnings))

    def test_env_file_is_merged_but_process_env_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('GEMINI_API_KEY="file-key-1234567"\nSYNTHETIC_SEED=7\n', encoding="utf-8")
            settings = load_settings(env_file=path, environ={"SYNTHETIC_SEED": "9"})
            self.assertEqual(settings.gemini_api_key, "file-key-1234567")
            self.assertEqual(settings.synthetic_seed, 9)

    def test_example_file_matches_defaults_and_is_loadable(self) -> None:
        example = PROJECT_ROOT / ".env.example"
        parsed = parse_dotenv(example.read_text(encoding="utf-8"))
        self.assertEqual(set(parsed), set(DEFAULTS), "keep .env.example and DEFAULTS in sync")
        self.assertEqual(parsed["GEMINI_API_KEY"], "")
        settings = load_settings(env_file=example, environ={})
        self.assertTrue(settings.offline_mode)
        self.assertEqual(settings.synthetic_records, int(DEFAULTS["SYNTHETIC_RECORDS"]))

    def test_settings_is_frozen(self) -> None:
        settings = Settings()
        with self.assertRaises(Exception):
            settings.gemini_model = "x"  # type: ignore[misc]


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_settings_cache()

    def tearDown(self) -> None:
        reset_settings_cache()

    def test_get_settings_caches_and_reloads(self) -> None:
        first = get_settings(env_file=None)
        second = get_settings(env_file=None)
        self.assertIs(first, second)
        third = get_settings(env_file=None, reload=True)
        self.assertIsNot(first, third)
        self.assertEqual(first, third)


class GitignoreTests(unittest.TestCase):
    def test_required_exclusions_present(self) -> None:
        text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        rules = {line.strip() for line in text if line.strip() and not line.startswith("#")}
        for required in (".env", "venv/", ".venv/", "__pycache__/", ".pytest_cache/", ".streamlit/",
                         "*.sqlite3", "*.db", "*.log", "!.env.example", "!.streamlit/config.toml"):
            self.assertIn(required, rules, required)


if __name__ == "__main__":
    unittest.main()
