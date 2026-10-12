"""Tests for ``src.llm_parser``. No network access is ever made.

Run with ``python -m unittest discover tests``.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

from google.genai import errors as genai_errors

from src.config import load_settings
from src.llm_parser import (
    SYNTHETIC_SCENARIOS,
    GeminiOSINTParser,
    ParsedThreat,
    ParserMode,
    RuleBasedOSINTParser,
    canonicalize_region,
    classify_api_error,
    demo_messages,
    match_scenario,
)

OFFLINE_SETTINGS = load_settings(env_file=None, environ={})
KEYED_SETTINGS = load_settings(env_file=None, environ={"GEMINI_API_KEY": "test-key-1234567890"})


class FakeModels:
    """Stand-in for ``client.models`` with scripted responses / exceptions."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("unexpected extra Gemini call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def fake_client(script: list[Any]) -> tuple[Any, FakeModels]:
    models = FakeModels(script)
    return SimpleNamespace(models=models), models


def response(payload: dict[str, Any] | str) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(text=text, parsed=None)


GOOD_PAYLOAD = {
    "threat_type": "uav_shahed",
    "source_region": "Херсонщина",
    "target_regions": ["Mykolaiv", "Одеська область", "Narnia"],
    "confidence_score": 0.93,
    "raw_summary": "Shahed group from Kherson towards Mykolaiv and Odesa.",
    "direction": "west",
    "estimated_count": 8,
}


class RegionTests(unittest.TestCase):
    def test_canonicalize_variants(self) -> None:
        cases = {
            "Харківщина": "Kharkiv Oblast",
            "kharkiv": "Kharkiv Oblast",
            "Kharkiv Oblast": "Kharkiv Oblast",
            "Київщина": "Kyiv Oblast",
            "Київ": "Kyiv City",
            "kyiv": "Kyiv City",
            "Сумщину": "Sumy Oblast",
            "Dnipro": "Dnipropetrovsk Oblast",
            "Кривий Ріг": "Dnipropetrovsk Oblast",
            "Бєлгород": "Belgorod (RU)",
            "акваторії Чорного моря": "Black Sea",
            "Crimea": "Crimea (occupied)",
            "Narnia": None,
            "": None,
            None: None,
        }
        for raw, expected in cases.items():
            self.assertEqual(canonicalize_region(raw), expected, raw)


class ParsedThreatValidationTests(unittest.TestCase):
    def test_canonicalises_and_derives_wave(self) -> None:
        threat = ParsedThreat(**GOOD_PAYLOAD)
        self.assertEqual(threat.threat_type, "uav_shahed")
        self.assertEqual(threat.wave_class, "slow")
        self.assertEqual(threat.source_region, "Kherson Oblast")
        self.assertEqual(threat.target_regions, ["Mykolaiv Oblast", "Odesa Oblast"])
        self.assertEqual(threat.unmapped_regions, ["Narnia"])
        self.assertEqual(threat.direction, "west")
        self.assertEqual(threat.estimated_count, 8)
        self.assertIsNotNone(threat.parsed_at.tzinfo)

    def test_slang_threat_type_and_clamping(self) -> None:
        threat = ParsedThreat(
            threat_type="Shahed", confidence_score=1.7, raw_summary="  x  " * 200,
            estimated_count=-3, direction="північно-західному", target_regions="Сумщина, Полтавщина",
        )
        self.assertEqual(threat.threat_type, "uav_shahed")
        self.assertEqual(threat.confidence_score, 1.0)
        self.assertLessEqual(len(threat.raw_summary), 280)
        self.assertIsNone(threat.estimated_count)
        self.assertEqual(threat.direction, "north-west")
        self.assertEqual(threat.target_regions, ["Sumy Oblast", "Poltava Oblast"])

    def test_garbage_values_do_not_raise(self) -> None:
        threat = ParsedThreat(threat_type=None, confidence_score="high", raw_summary=None,
                              source_region="Mars", target_regions=None)
        self.assertEqual(threat.threat_type, "unknown")
        self.assertEqual(threat.wave_class, "unknown")
        self.assertEqual(threat.confidence_score, 0.0)
        self.assertEqual(threat.raw_summary, "Unparsed air-threat report")
        self.assertIsNone(threat.source_region)
        self.assertEqual(threat.unmapped_regions, ["Mars"])

    def test_external_origin_never_a_target(self) -> None:
        threat = ParsedThreat(threat_type="ballistic", target_regions=["Belgorod", "Kharkiv"])
        self.assertEqual(threat.target_regions, ["Kharkiv Oblast"])

    def test_frozen_and_json(self) -> None:
        threat = ParsedThreat(**GOOD_PAYLOAD)
        with self.assertRaises(Exception):
            threat.threat_type = "aviation"  # type: ignore[misc]
        self.assertIn('"wave_class":"slow"', threat.to_json())


class RuleBasedParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RuleBasedOSINTParser()

    def test_shahed_route_ukrainian(self) -> None:
        out = self.parser.parse("⚠️ Сумщина: 6 мопедів курсом на Полтавщину, швидкість ~150 км/год")
        self.assertEqual(out.threat_type, "uav_shahed")
        self.assertEqual(out.wave_class, "slow")
        self.assertEqual(out.source_region, "Sumy Oblast")
        self.assertEqual(out.target_regions, ["Poltava Oblast"])
        self.assertEqual(out.estimated_count, 6)
        self.assertIn("150 km/h", out.raw_summary)
        self.assertEqual(out.parser, "rules")
        self.assertGreaterEqual(out.confidence_score, 0.7)
        self.assertLessEqual(out.confidence_score, RuleBasedOSINTParser.MAX_CONFIDENCE)

    def test_ballistic_from_external_origin(self) -> None:
        out = self.parser.parse("Пуски балістики з Бєлгорода на Харків!")
        self.assertEqual(out.threat_type, "ballistic_missile")
        self.assertEqual(out.wave_class, "fast")
        self.assertEqual(out.source_region, "Belgorod (RU)")
        self.assertEqual(out.target_regions, ["Kharkiv Oblast"])

    def test_bomber_launch_vs_takeoff(self) -> None:
        launch = self.parser.parse("Пуск крилатих ракет з Ту-95 над Каспієм")
        takeoff = self.parser.parse("Зліт Ту-95 з Енгельса")
        self.assertEqual(launch.threat_type, "cruise_missile")
        self.assertEqual(launch.source_region, "Caspian Sea")
        self.assertEqual(takeoff.threat_type, "aviation")
        self.assertEqual(takeoff.source_region, "Engels (RU)")

    def test_arrow_targets_and_count(self) -> None:
        out = self.parser.parse("Калібри з Чорного моря → Одещина, Миколаївщина. 4 ракети.")
        self.assertEqual(out.threat_type, "cruise_missile")
        self.assertEqual(out.source_region, "Black Sea")
        self.assertEqual(out.target_regions, ["Odesa Oblast", "Mykolaiv Oblast"])
        self.assertEqual(out.estimated_count, 4)

    def test_english_and_russian(self) -> None:
        english = self.parser.parse("3 shahed drones from Crimea heading north-west towards Mykolaiv")
        self.assertEqual(english.threat_type, "uav_shahed")
        self.assertEqual(english.source_region, "Crimea (occupied)")
        self.assertEqual(english.target_regions, ["Mykolaiv Oblast"])
        self.assertEqual(english.direction, "north-west")
        self.assertEqual(english.estimated_count, 3)
        # Convention: a bomber/MiG *takeoff* is an aviation event even when the
        # message names the Kinzhal it may carry (matches the synthetic scenario).
        russian = self.parser.parse("Взлет МиГ-31 из Саваслейки, угроза Кинжала")
        self.assertEqual(russian.threat_type, "aviation")
        self.assertEqual(russian.wave_class, "fast")
        self.assertEqual(russian.source_region, "Savasleyka (RU)")
        launch = self.parser.parse("Пуск Кинджала з МіГ-31К по Київщині")
        self.assertEqual(launch.threat_type, "ballistic_missile")
        self.assertEqual(launch.target_regions, ["Kyiv Oblast"])

    def test_recon_beats_generic_uav_and_single_region_is_target(self) -> None:
        out = self.parser.parse("Розвідувальний БпЛА типу Орлан над Дніпропетровщиною")
        self.assertEqual(out.threat_type, "uav_recon")
        self.assertIsNone(out.source_region)
        self.assertEqual(out.target_regions, ["Dnipropetrovsk Oblast"])

    def test_kab_and_artillery(self) -> None:
        self.assertEqual(self.parser.parse("КАБи по Сумщині").threat_type, "guided_bomb")
        self.assertEqual(self.parser.parse("Обстріл з РСЗВ Херсона").threat_type, "artillery")

    def test_unknown_and_empty(self) -> None:
        vague = self.parser.parse("Увага! Будьте обережні.")
        self.assertEqual(vague.threat_type, "unknown")
        self.assertEqual(vague.target_regions, [])
        self.assertLess(vague.confidence_score, 0.4)
        empty = self.parser.parse("")
        self.assertEqual(empty.confidence_score, 0.0)
        self.assertEqual(empty.threat_type, "unknown")


class ScenarioTests(unittest.TestCase):
    def test_all_scenarios_validate_and_match(self) -> None:
        names = {s.name for s in SYNTHETIC_SCENARIOS}
        self.assertEqual(len(names), len(SYNTHETIC_SCENARIOS))
        for scenario in SYNTHETIC_SCENARIOS:
            threat = scenario.to_threat()
            self.assertEqual(threat.parser, "synthetic")
            self.assertEqual(threat.threat_type, scenario.payload["threat_type"])
            self.assertEqual(match_scenario(scenario.message), scenario, scenario.name)
            self.assertEqual(threat.unmapped_regions, [], scenario.name)
        self.assertEqual(len(demo_messages()), len(SYNTHETIC_SCENARIOS))

    def test_fuzzy_and_negative_matches(self) -> None:
        fuzzy = match_scenario("знову зліт міг-31 з саваслейки, увага")
        self.assertIsNotNone(fuzzy)
        self.assertEqual(fuzzy.name, "mig31k_takeoff_savasleyka")  # type: ignore[union-attr]
        self.assertIsNone(match_scenario("Сумщина: 6 мопедів курсом на Полтавщину"))
        self.assertIsNone(match_scenario(""))

    def test_rules_agree_with_scenario_types(self) -> None:
        rules = RuleBasedOSINTParser()
        for scenario in SYNTHETIC_SCENARIOS:
            self.assertEqual(rules.parse(scenario.message).threat_type,
                             scenario.payload["threat_type"], scenario.name)


class ErrorClassificationTests(unittest.TestCase):
    def test_permanent_vs_transient(self) -> None:
        forbidden = genai_errors.ClientError(403, {"error": {"message": "API key not valid",
                                                             "status": "PERMISSION_DENIED"}})
        unauthorised = genai_errors.ClientError(401, {"error": {"message": "unauthenticated"}})
        bad_key = genai_errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key.",
                                                           "status": "INVALID_ARGUMENT"}})
        not_found = genai_errors.ClientError(404, {"error": {"message": "model not found"}})
        rate = genai_errors.ClientError(429, {"error": {"message": "quota exceeded"}})
        server = genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
        for exc in (forbidden, unauthorised, bad_key, not_found, ImportError("google.genai")):
            self.assertEqual(classify_api_error(exc), "permanent", exc)
        for exc in (rate, server, TimeoutError("t/o"), ConnectionError("down"), ValueError("bad json")):
            self.assertEqual(classify_api_error(exc), "transient", exc)


class GeminiParserTests(unittest.TestCase):
    MSG = "Сумщина: 6 мопедів курсом на Полтавщину"

    def test_missing_key_starts_offline(self) -> None:
        parser = GeminiOSINTParser(settings=OFFLINE_SETTINGS,
                                   client_factory=lambda key: self.fail("must not build a client"))
        self.assertEqual(parser.mode, ParserMode.OFFLINE)
        self.assertIn("GEMINI_API_KEY", parser.offline_reason or "")
        out = parser.parse(self.MSG)
        self.assertEqual(out.parser, "rules")
        self.assertEqual(out.threat_type, "uav_shahed")
        self.assertEqual(parser.status()["stats"], {"rules": 1})

    def test_llm_disabled_flag_and_force_offline(self) -> None:
        disabled = load_settings(env_file=None, environ={"GEMINI_API_KEY": "k" * 20, "LLM_ENABLED": "false"})
        self.assertEqual(GeminiOSINTParser(settings=disabled).mode, ParserMode.OFFLINE)
        forced = GeminiOSINTParser(settings=KEYED_SETTINGS, force_offline=True)
        self.assertEqual(forced.mode, ParserMode.OFFLINE)
        self.assertEqual(forced.parse(self.MSG).parser, "rules")

    def test_offline_returns_synthetic_payload_for_demo_message(self) -> None:
        parser = GeminiOSINTParser(settings=OFFLINE_SETTINGS)
        out = parser.parse(demo_messages()[0])
        self.assertEqual(out.parser, "synthetic")
        self.assertEqual(out.threat_type, "uav_shahed")
        self.assertEqual(out.estimated_count, 12)
        plain = GeminiOSINTParser(settings=OFFLINE_SETTINGS, use_synthetic_scenarios=False)
        self.assertEqual(plain.parse(demo_messages()[0]).parser, "rules")

    def test_successful_gemini_call(self) -> None:
        client, models = fake_client([response(GOOD_PAYLOAD)])
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client)
        self.assertEqual(parser.mode, ParserMode.GEMINI)
        out = parser.parse(self.MSG)
        self.assertEqual(out.parser, "gemini")
        self.assertEqual(out.model, KEYED_SETTINGS.gemini_model)
        self.assertEqual(out.source_region, "Kherson Oblast")
        self.assertEqual(out.target_regions, ["Mykolaiv Oblast", "Odesa Oblast"])
        self.assertEqual(out.wave_class, "slow")
        self.assertEqual(len(models.calls), 1)
        call = models.calls[0]
        self.assertEqual(call["model"], KEYED_SETTINGS.gemini_model)
        self.assertIn(self.MSG, call["contents"])
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertEqual(parser.status()["stats"], {"gemini": 1})

    def test_parsed_attribute_and_code_fences(self) -> None:
        fenced = response("```json\n" + json.dumps(GOOD_PAYLOAD) + "\n```")
        structured = SimpleNamespace(text=None, parsed=SimpleNamespace(model_dump=lambda: dict(GOOD_PAYLOAD)))
        client, _ = fake_client([fenced, structured])
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client)
        self.assertEqual(parser.parse(self.MSG).parser, "gemini")
        self.assertEqual(parser.parse(self.MSG).parser, "gemini")

    def test_malformed_json_falls_back_but_stays_online(self) -> None:
        client, models = fake_client([response("not json at all"), response("still not json")])
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client,
                                   max_attempts=2, sleep=lambda s: None)
        out = parser.parse(self.MSG)
        self.assertEqual(out.parser, "rules")
        self.assertEqual(out.threat_type, "uav_shahed")
        self.assertEqual(parser.mode, ParserMode.GEMINI)  # transient: keep trying next time
        self.assertEqual(len(models.calls), 2)  # retried once
        self.assertIn("JSONDecodeError", parser.last_error or "")

    def test_auth_error_disables_gemini_permanently(self) -> None:
        forbidden = genai_errors.ClientError(403, {"error": {"message": "API key not valid"}})
        client, models = fake_client([forbidden])
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client)
        first = parser.parse(self.MSG)
        self.assertEqual(first.parser, "rules")
        self.assertEqual(parser.mode, ParserMode.OFFLINE)
        self.assertIn("API key not valid", parser.offline_reason or "")
        parser.parse(self.MSG)  # no further API calls
        self.assertEqual(len(models.calls), 1)

    def test_transient_retry_then_success(self) -> None:
        client, models = fake_client([TimeoutError("read timeout"), response(GOOD_PAYLOAD)])
        slept: list[float] = []
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client,
                                   max_attempts=2, sleep=slept.append)
        out = parser.parse(self.MSG)
        self.assertEqual(out.parser, "gemini")
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(slept, [0.5])
        self.assertEqual(parser.status()["consecutive_failures"], 0)

    def test_circuit_breaker_opens_after_repeated_failures(self) -> None:
        outage = [genai_errors.ServerError(503, {"error": {"message": "overloaded"}}) for _ in range(6)]
        client, models = fake_client(outage)
        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=lambda key: client,
                                   max_attempts=1, max_consecutive_failures=3)
        for _ in range(5):
            out = parser.parse(self.MSG)
            self.assertEqual(out.parser, "rules")
        self.assertEqual(parser.mode, ParserMode.OFFLINE)
        self.assertIn("circuit breaker", parser.offline_reason or "")
        self.assertEqual(len(models.calls), 3)  # stopped calling after the breaker opened

    def test_client_construction_failure_degrades(self) -> None:
        def broken_factory(key: str) -> Any:
            raise ImportError("No module named google.genai")

        parser = GeminiOSINTParser(settings=KEYED_SETTINGS, client_factory=broken_factory)
        out = parser.parse(self.MSG)
        self.assertEqual(out.parser, "rules")
        self.assertEqual(parser.mode, ParserMode.OFFLINE)

    def test_parse_many_and_empty(self) -> None:
        parser = GeminiOSINTParser(settings=OFFLINE_SETTINGS)
        results = parser.parse_many([self.MSG, "", demo_messages()[1]])
        self.assertEqual([r.parser for r in results], ["rules", "rules", "synthetic"])
        self.assertEqual(results[1].threat_type, "unknown")
        self.assertEqual(parser.status()["stats"]["empty"], 1)


if __name__ == "__main__":
    unittest.main()
