"""Gemini-backed parser for informal Telegram OSINT air-threat updates.

Monitoring channels post terse, slang-heavy messages such as::

    ⚠️ Сумщина: 6 мопедів курсом на Полтавщину, швидкість ~150 км/год
    Зліт МіГ-31К з Саваслейки. Загроза балістики по всій Україні!
    Пуски Калібрів з Чорного моря → Одещина, Миколаївщина

This module turns them into a validated :class:`ParsedThreat` record that the
cascade engine can consume. It has two engines behind one interface:

* :class:`GeminiOSINTParser` calls Google Gemini through the official
  ``google-genai`` SDK with a strict JSON response schema.
* :class:`RuleBasedOSINTParser` is a dependency-free keyword/heuristic parser
  that understands Ukrainian, Russian and English monitoring slang.

The Gemini parser **always** degrades to the rule-based parser (or to a
pre-defined synthetic payload for known demo scenarios) when the API key is
missing or invalid, the SDK is not installed, the network is down, the model
returns malformed output, or repeated transient failures trip the circuit
breaker. Downstream code never has to handle a crash from this module.

Output schema
-------------
``threat_type``       canonical key from :data:`src.loader.THREAT_TYPES` or
                      ``"unknown"``
``wave_class``        ``"fast"`` (ballistic / cruise / aviation / bombs /
                      artillery) or ``"slow"`` (UAV / Shahed) or ``"unknown"``
``source_region``     launch / origin area: a Ukrainian oblast or a known
                      external origin such as ``"Belgorod (RU)"``; ``None`` if
                      not stated
``target_regions``    projected impacted oblasts (canonical names)
``confidence_score``  ``0.0`` – ``1.0``
``raw_summary``       short normalised English description
plus auxiliary ``direction``, ``estimated_count``, ``unmapped_regions``,
``parser`` (``"gemini"`` / ``"rules"`` / ``"synthetic"``), ``model`` and
``parsed_at``.

Example
-------
>>> parser = GeminiOSINTParser()          # offline automatically if no key
>>> result = parser.parse("Сумщина: 6 мопедів курсом на Полтавщину")
>>> result.threat_type, result.wave_class, result.target_regions
('uav_shahed', 'slow', ['Poltava Oblast'])
"""

from __future__ import annotations

import json
import logging
import math
import re
import time as _time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Final, Iterable, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.config import Settings, get_settings
from src.loader import THREAT_TYPES, UKRAINIAN_REGIONS, ThreatWave, normalize_threat_type

__all__ = [
    "EXTERNAL_ORIGINS",
    "GeminiOSINTParser",
    "GeminiThreatSchema",
    "ParsedThreat",
    "ParserMode",
    "RuleBasedOSINTParser",
    "SYNTHETIC_SCENARIOS",
    "SyntheticScenario",
    "canonicalize_region",
    "classify_api_error",
    "demo_messages",
    "match_scenario",
]

logger = logging.getLogger(__name__)

ThreatTypeLiteral = Literal[
    "ballistic_missile",
    "cruise_missile",
    "aviation",
    "guided_bomb",
    "artillery",
    "uav_shahed",
    "uav_recon",
    "unknown",
]
WaveLiteral = Literal["fast", "slow", "unknown"]
ParserLiteral = Literal["gemini", "rules", "synthetic"]
DirectionLiteral = Literal[
    "north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"
]

# --------------------------------------------------------------------------- #
# Region vocabulary
# --------------------------------------------------------------------------- #
EXTERNAL_ORIGINS: Final[Sequence[str]] = (
    "Belgorod (RU)",
    "Kursk (RU)",
    "Bryansk (RU)",
    "Voronezh (RU)",
    "Rostov (RU)",
    "Oryol (RU)",
    "Lipetsk (RU)",
    "Engels (RU)",
    "Savasleyka (RU)",
    "Shaikovka (RU)",
    "Millerovo (RU)",
    "Primorsko-Akhtarsk (RU)",
    "Crimea (occupied)",
    "Black Sea",
    "Sea of Azov",
    "Caspian Sea",
    "Belarus",
)
"""Launch areas outside government-controlled Ukraine that may appear as
``source_region``."""

# Lower-cased stems (Ukrainian / Russian / English) -> canonical area name.
# Longer stems are matched first so "київщин" wins over "київ".
_REGION_ALIASES: Final[dict[str, str]] = {
    # Kyiv Oblast must precede Kyiv City (shared prefix).
    "київщин": "Kyiv Oblast", "киевщин": "Kyiv Oblast", "київська обл": "Kyiv Oblast",
    "київської обл": "Kyiv Oblast", "kyiv oblast": "Kyiv Oblast", "kyiv region": "Kyiv Oblast",
    "бориспіл": "Kyiv Oblast", "біла церква": "Kyiv Oblast", "бровар": "Kyiv Oblast",
    "вишгород": "Kyiv Oblast", "обухів": "Kyiv Oblast", "фастів": "Kyiv Oblast",
    "київ": "Kyiv City", "киев": "Kyiv City", "kyiv": "Kyiv City", "kiev": "Kyiv City",
    "харків": "Kharkiv Oblast", "харьков": "Kharkiv Oblast", "kharkiv": "Kharkiv Oblast",
    "kharkov": "Kharkiv Oblast", "чугуїв": "Kharkiv Oblast", "ізюм": "Kharkiv Oblast",
    "одещин": "Odesa Oblast", "одесщин": "Odesa Oblast", "одеськ": "Odesa Oblast",
    "одесск": "Odesa Oblast", "одес": "Odesa Oblast", "odesa": "Odesa Oblast",
    "odessa": "Odesa Oblast", "ізмаїл": "Odesa Oblast", "измаил": "Odesa Oblast",
    "дніпропетров": "Dnipropetrovsk Oblast", "днепропетров": "Dnipropetrovsk Oblast",
    "дніпр": "Dnipropetrovsk Oblast", "днепр": "Dnipropetrovsk Oblast", "dnipro": "Dnipropetrovsk Oblast",
    "dnepr": "Dnipropetrovsk Oblast", "кривий ріг": "Dnipropetrovsk Oblast",
    "кривой рог": "Dnipropetrovsk Oblast", "kryvyi rih": "Dnipropetrovsk Oblast",
    "павлоград": "Dnipropetrovsk Oblast", "нікопол": "Dnipropetrovsk Oblast",
    "запоріж": "Zaporizhzhia Oblast", "запорож": "Zaporizhzhia Oblast", "zaporizh": "Zaporizhzhia Oblast",
    "zaporozh": "Zaporizhzhia Oblast",
    "миколаїв": "Mykolaiv Oblast", "николаев": "Mykolaiv Oblast", "mykolaiv": "Mykolaiv Oblast",
    "nikolaev": "Mykolaiv Oblast",
    "херсон": "Kherson Oblast", "kherson": "Kherson Oblast",
    "сумщин": "Sumy Oblast", "суми": "Sumy Oblast", "сумы": "Sumy Oblast", "сумськ": "Sumy Oblast",
    "sumy": "Sumy Oblast", "шостк": "Sumy Oblast", "конотоп": "Sumy Oblast", "охтирк": "Sumy Oblast",
    "чернігів": "Chernihiv Oblast", "чернигов": "Chernihiv Oblast", "chernihiv": "Chernihiv Oblast",
    "chernigov": "Chernihiv Oblast", "ніжин": "Chernihiv Oblast",
    "полтав": "Poltava Oblast", "poltava": "Poltava Oblast", "кременчу": "Poltava Oblast",
    "миргород": "Poltava Oblast",
    "черкас": "Cherkasy Oblast", "черкащ": "Cherkasy Oblast", "cherkas": "Cherkasy Oblast",
    "умань": "Cherkasy Oblast",
    "кіровоград": "Kirovohrad Oblast", "кировоград": "Kirovohrad Oblast", "кропивниц": "Kirovohrad Oblast",
    "kirovohrad": "Kirovohrad Oblast", "kropyvnyts": "Kirovohrad Oblast",
    "вінниц": "Vinnytsia Oblast", "вінничч": "Vinnytsia Oblast", "винниц": "Vinnytsia Oblast",
    "винничч": "Vinnytsia Oblast", "vinnyts": "Vinnytsia Oblast",
    "житомир": "Zhytomyr Oblast", "zhytomyr": "Zhytomyr Oblast", "zhitomir": "Zhytomyr Oblast",
    "хмельниц": "Khmelnytskyi Oblast", "хмельничч": "Khmelnytskyi Oblast",
    "khmelnyts": "Khmelnytskyi Oblast", "старокостянтин": "Khmelnytskyi Oblast",
    "рівнен": "Rivne Oblast", "ровен": "Rivne Oblast", "rivne": "Rivne Oblast", "rovno": "Rivne Oblast",
    "волин": "Volyn Oblast", "луцьк": "Volyn Oblast", "луцк": "Volyn Oblast", "volyn": "Volyn Oblast",
    "lutsk": "Volyn Oblast",
    "львів": "Lviv Oblast", "львов": "Lviv Oblast", "lviv": "Lviv Oblast", "lvov": "Lviv Oblast",
    "терноп": "Ternopil Oblast", "ternopil": "Ternopil Oblast",
    "івано-франк": "Ivano-Frankivsk Oblast", "ивано-франк": "Ivano-Frankivsk Oblast",
    "франківськ": "Ivano-Frankivsk Oblast", "ivano-frank": "Ivano-Frankivsk Oblast",
    "чернівц": "Chernivtsi Oblast", "черновц": "Chernivtsi Oblast", "буковин": "Chernivtsi Oblast",
    "chernivtsi": "Chernivtsi Oblast",
    "закарпат": "Zakarpattia Oblast", "ужгород": "Zakarpattia Oblast", "zakarpat": "Zakarpattia Oblast",
    "uzhhorod": "Zakarpattia Oblast", "transcarpath": "Zakarpattia Oblast",
    "донеч": "Donetsk Oblast", "донецьк": "Donetsk Oblast", "донецк": "Donetsk Oblast",
    "donetsk": "Donetsk Oblast", "краматор": "Donetsk Oblast", "слов'ян": "Donetsk Oblast",
    "покровськ": "Donetsk Oblast",
    "луган": "Luhansk Oblast", "luhansk": "Luhansk Oblast", "lugansk": "Luhansk Oblast",
    # External origins.
    "бєлгород": "Belgorod (RU)", "белгород": "Belgorod (RU)", "belgorod": "Belgorod (RU)",
    "курськ": "Kursk (RU)", "курск": "Kursk (RU)", "kursk": "Kursk (RU)",
    "брянськ": "Bryansk (RU)", "брянск": "Bryansk (RU)", "bryansk": "Bryansk (RU)",
    "воронеж": "Voronezh (RU)", "voronezh": "Voronezh (RU)",
    "ростов": "Rostov (RU)", "rostov": "Rostov (RU)",
    "орел": "Oryol (RU)", "орловськ": "Oryol (RU)", "орловск": "Oryol (RU)", "oryol": "Oryol (RU)",
    "липецьк": "Lipetsk (RU)", "липецк": "Lipetsk (RU)", "lipetsk": "Lipetsk (RU)",
    "енгельс": "Engels (RU)", "энгельс": "Engels (RU)", "engels": "Engels (RU)",
    "саваслейк": "Savasleyka (RU)", "savasleyka": "Savasleyka (RU)",
    "шайковк": "Shaikovka (RU)", "shaikovka": "Shaikovka (RU)",
    "міллерово": "Millerovo (RU)", "миллерово": "Millerovo (RU)", "millerovo": "Millerovo (RU)",
    "приморсько-ахтарськ": "Primorsko-Akhtarsk (RU)", "приморско-ахтарск": "Primorsko-Akhtarsk (RU)",
    "akhtarsk": "Primorsko-Akhtarsk (RU)",
    "крим": "Crimea (occupied)", "крым": "Crimea (occupied)", "crimea": "Crimea (occupied)",
    "джанко": "Crimea (occupied)", "севастопол": "Crimea (occupied)", "саки": "Crimea (occupied)",
    "чорного моря": "Black Sea", "чорне море": "Black Sea", "черного моря": "Black Sea",
    "black sea": "Black Sea",
    "азовськ": "Sea of Azov", "азовск": "Sea of Azov", "azov": "Sea of Azov",
    "каспі": "Caspian Sea", "каспий": "Caspian Sea", "caspian": "Caspian Sea",
    "білорус": "Belarus", "беларус": "Belarus", "belarus": "Belarus",
}
_ALIASES_BY_LENGTH: Final[Sequence[tuple[str, str]]] = tuple(
    sorted(_REGION_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)
)
_CANONICAL_LOOKUP: Final[dict[str, str]] = {
    **{name.lower(): name for name in UKRAINIAN_REGIONS},
    **{name.lower(): name for name in EXTERNAL_ORIGINS},
    # Suffix-less shortcuts ("kharkiv" -> Kharkiv Oblast). Kyiv is excluded:
    # a bare "kyiv" means the city, and "Kyiv Oblast" must be spelled out.
    **{name.lower().replace(" oblast", ""): name for name in UKRAINIAN_REGIONS
       if name != "Kyiv Oblast"},
}


def canonicalize_region(name: object) -> Optional[str]:
    """Map a free-form region mention to a canonical area name.

    Accepts canonical names, names without the ``Oblast`` suffix, Ukrainian /
    Russian / English spellings and adjectival forms (``"Сумщина"``), major
    city names and known external launch areas.

    Args:
        name: Raw mention.

    Returns:
        A member of :data:`src.loader.UKRAINIAN_REGIONS` or
        :data:`EXTERNAL_ORIGINS`, or ``None`` when it cannot be resolved.

    Examples:
        >>> canonicalize_region("Харківщина")
        'Kharkiv Oblast'
        >>> canonicalize_region("kyiv")
        'Kyiv City'
        >>> canonicalize_region("Narnia") is None
        True
    """
    if name is None:
        return None
    text = str(name).strip().lower().replace("’", "'")
    if not text:
        return None
    if text in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[text]
    for stem, canonical in _ALIASES_BY_LENGTH:
        if stem in text:
            return canonical
    return None


# --------------------------------------------------------------------------- #
# Validated output model
# --------------------------------------------------------------------------- #
class GeminiThreatSchema(BaseModel):
    """Response schema handed to Gemini (kept flat and permissive on purpose).

    Region fields are free text here; :class:`ParsedThreat` canonicalises
    them afterwards so the model may answer in any language.
    """

    threat_type: ThreatTypeLiteral = Field(description="Canonical threat category.")
    source_region: Optional[str] = Field(
        default=None, description="Launch / origin area if stated, else null."
    )
    target_regions: list[str] = Field(
        default_factory=list, description="Oblasts projected to be affected, in order of mention."
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="Parser confidence 0-1.")
    raw_summary: str = Field(description="One-sentence normalised English summary.")
    direction: Optional[DirectionLiteral] = Field(
        default=None, description="Heading of the threat if stated."
    )
    estimated_count: Optional[int] = Field(
        default=None, description="Number of objects if stated, else null."
    )


class ParsedThreat(BaseModel):
    """Validated, canonicalised OSINT threat record.

    Attributes:
        threat_type: Canonical key from :data:`src.loader.THREAT_TYPES` or
            ``"unknown"``.
        wave_class: Derived kinematic class (``fast`` / ``slow`` / ``unknown``).
        source_region: Canonical origin area or ``None``.
        target_regions: Canonical projected target oblasts (deduplicated).
        confidence_score: Confidence clamped to ``[0, 1]``.
        raw_summary: Normalised description, at most 280 characters.
        direction: Normalised compass heading or ``None``.
        estimated_count: Positive object count or ``None``.
        unmapped_regions: Mentions that could not be canonicalised (audit).
        parser: Engine that produced the record.
        model: Gemini model id when ``parser == "gemini"``.
        source_excerpt: First 160 characters of the input message.
        parsed_at: UTC timestamp of parsing.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    threat_type: ThreatTypeLiteral = "unknown"
    wave_class: WaveLiteral = "unknown"
    source_region: Optional[str] = None
    target_regions: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    raw_summary: str = "Unparsed air-threat report"
    direction: Optional[DirectionLiteral] = None
    estimated_count: Optional[int] = None
    unmapped_regions: list[str] = Field(default_factory=list)
    parser: ParserLiteral = "rules"
    model: Optional[str] = None
    source_excerpt: str = ""
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _canonicalise(cls, data: Any) -> Any:
        """Normalise threat type, wave class and region names before typing."""
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        unmapped: list[str] = [str(x) for x in (payload.get("unmapped_regions") or [])]

        threat = normalize_threat_type(payload.get("threat_type"))
        payload["threat_type"] = threat
        payload["wave_class"] = THREAT_TYPES.get(threat, ThreatWave.UNKNOWN).value

        raw_source = payload.get("source_region")
        source = canonicalize_region(raw_source) if raw_source not in (None, "") else None
        if raw_source not in (None, "") and source is None:
            unmapped.append(str(raw_source))
        payload["source_region"] = source

        targets: list[str] = []
        raw_targets = payload.get("target_regions") or []
        if isinstance(raw_targets, str):
            raw_targets = re.split(r"[;,/]|\s+(?:and|та|и)\s+", raw_targets)
        for item in raw_targets:
            canonical = canonicalize_region(item)
            if canonical is None:
                if str(item).strip():
                    unmapped.append(str(item).strip())
            elif canonical not in targets and canonical in UKRAINIAN_REGIONS:
                targets.append(canonical)
        payload["target_regions"] = targets
        payload["unmapped_regions"] = list(dict.fromkeys(unmapped))

        direction = payload.get("direction")
        payload["direction"] = _normalise_direction(direction) if direction else None
        return payload

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return min(1.0, max(0.0, number))

    @field_validator("estimated_count", mode="before")
    @classmethod
    def _positive_count(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @field_validator("raw_summary", mode="before")
    @classmethod
    def _tidy_summary(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return "Unparsed air-threat report"
        return text[:277] + "…" if len(text) > 280 else text

    @field_validator("source_excerpt", mode="before")
    @classmethod
    def _tidy_excerpt(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()[:160]

    @field_validator("parsed_at", mode="after")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    def to_json(self) -> str:
        """Serialise to a compact JSON string."""
        return self.model_dump_json()


# --------------------------------------------------------------------------- #
# Rule-based parser
# --------------------------------------------------------------------------- #
_THREAT_KEYWORDS: Final[Mapping[str, tuple[str, ...]]] = {
    "ballistic_missile": (
        "баліст", "ballistic", "іскандер", "искандер", "iskander", "кинджал", "кинжал", "kinzhal",
        "точка-у", "kn-23", "швидкісн", "скоростн", "high-speed",
    ),
    "cruise_missile": (
        "крилат", "крылат", "cruise", "калібр", "калибр", "kalibr", "х-101", "x-101", "х-555",
        "х-22", "х-32", "х-59", "х-69", "онікс", "оникс", "oniks", "циркон", "zircon",
    ),
    "guided_bomb": ("каб", "kab", "керован", "авіабомб", "авиабомб", "умпк", "glide bomb", "guided bomb"),
    "uav_recon": ("розвід", "развед", "орлан", "zala", "supercam", "recon", "мерлін", "мерлин"),
    "uav_shahed": (
        "шахед", "shahed", "мопед", "герань", "герані", "geran", "бпла", "дрон", "drone", "uav",
        "ударн",
    ),
    "aviation": (
        "авіац", "авиац", "aviation", "міг-31", "миг-31", "mig-31", "су-34", "su-34", "су-35",
        "su-35", "ту-95", "tu-95", "ту-160", "tu-160", "ту-22", "tu-22", "зліт", "взлет", "злет",
        "takeoff", "take-off", "бомбардувальник", "bomber", "літак", "самолет",
    ),
    "artillery": ("артилер", "artiller", "рсзв", "mlrs", "обстріл", "обстрел", "shelling"),
}
_THREAT_PRIORITY: Final[Sequence[str]] = (
    "ballistic_missile", "cruise_missile", "guided_bomb", "uav_recon", "uav_shahed", "aviation",
    "artillery",
)
_THREAT_LABELS: Final[Mapping[str, str]] = {
    "ballistic_missile": "Ballistic missile threat",
    "cruise_missile": "Cruise missile launch",
    "aviation": "Tactical/strategic aviation activity",
    "guided_bomb": "Guided aerial bomb threat",
    "artillery": "Artillery / MLRS shelling",
    "uav_shahed": "Shahed-type strike UAVs",
    "uav_recon": "Reconnaissance UAV",
    "unknown": "Unclassified air threat",
}
_LAUNCH_WORDS: Final[tuple[str, ...]] = ("пуск", "пуски", "launch", "запуск")

_SOURCE_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:з|із|зі|из|с|от|from)\s*$|з боку\s*$|з напрямку\s*$|з території\s*$|"
    r"зліт з\s*$|launched from\s*$|акваторі[їи]\s*$)",
    re.IGNORECASE,
)
_TARGET_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:курс(?:ом)?\s+на|у напрямку|в напрямку|у бік|в бік|в сторону|прямує на|рухається на|"
    r"летять на|летить на|летят на|через|towards?|heading (?:to|for)|\bto\b|\bпо\b|\bна\b|→|->|=>)"
    r"\s*$",
    re.IGNORECASE,
)
_DIRECTION_PATTERNS: Final[Sequence[tuple[re.Pattern[str], str]]] = (
    (re.compile(r"північно[-\s]?захід|северо[-\s]?запад|north[-\s]?west|\bnw\b", re.I), "north-west"),
    (re.compile(r"північно[-\s]?схід|северо[-\s]?восток|north[-\s]?east|\bne\b", re.I), "north-east"),
    (re.compile(r"південно[-\s]?захід|юго[-\s]?запад|south[-\s]?west|\bsw\b", re.I), "south-west"),
    (re.compile(r"південно[-\s]?схід|юго[-\s]?восток|south[-\s]?east|\bse\b", re.I), "south-east"),
    (re.compile(r"північ|север|north", re.I), "north"),
    (re.compile(r"південь|південн|юг|южн|south", re.I), "south"),
    (re.compile(r"захід|західн|запад|west", re.I), "west"),
    (re.compile(r"схід|східн|восток|east", re.I), "east"),
)
_COUNT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d.,-])(\d{1,3})\s*(?:х|x|×|од\.?|шт\.?)?\s*"
    r"(?:шахед|мопед|бпла|дрон|ракет|балістич|крилат|каб|uav|drone|missile|shahed|герані|геран|"
    r"ударн|об'єкт|цілей|ціл[іь])",
    re.IGNORECASE,
)
_SPEED_RE: Final[re.Pattern[str]] = re.compile(r"(\d{2,4})\s*км/?год|(\d{2,4})\s*km/?h", re.I)


def _normalise_direction(value: object) -> Optional[str]:
    """Map a free-form heading to one of eight compass directions."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    for pattern, label in _DIRECTION_PATTERNS:
        if pattern.search(text):
            return label
    return None


@dataclass(frozen=True)
class _RegionMention:
    """A region reference located inside a message."""

    name: str
    start: int
    end: int
    external: bool


class RuleBasedOSINTParser:
    """Dependency-free heuristic parser for monitoring-channel slang.

    It recognises threat keywords in Ukrainian, Russian and English, locates
    region mentions, and uses nearby prepositions / arrows to decide whether
    a region is an origin (``з Бєлгорода``, ``from Crimea``) or a target
    (``курсом на Полтавщину``, ``→ Одещина``). Confidence is capped at
    ``0.85`` because heuristics never see context the model would.
    """

    MAX_CONFIDENCE: Final[float] = 0.85

    def parse(self, text: str) -> ParsedThreat:
        """Parse one message.

        Args:
            text: Raw Telegram message.

        Returns:
            A :class:`ParsedThreat` with ``parser == "rules"``. Never raises.
        """
        clean = re.sub(r"\s+", " ", (text or "")).strip()
        lowered = clean.lower().replace("’", "'")
        threat = self._detect_threat(lowered)
        mentions = self._find_regions(lowered)
        source, targets = self._split_source_targets(lowered, mentions)
        direction = self._detect_direction(lowered)
        count = self._detect_count(lowered)
        speed = self._detect_speed(lowered)
        confidence = self._confidence(threat, source, targets, direction, count, bool(clean))
        summary = self._summary(threat, source, targets, direction, count, speed)
        return ParsedThreat(
            threat_type=threat,
            source_region=source,
            target_regions=targets,
            confidence_score=confidence,
            raw_summary=summary,
            direction=direction,
            estimated_count=count,
            parser="rules",
            source_excerpt=clean,
        )

    # ---- detectors ------------------------------------------------------ #
    @staticmethod
    def _detect_threat(lowered: str) -> str:
        scores: Counter[str] = Counter()
        for threat, keywords in _THREAT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lowered:
                    scores[threat] += 1
        if not scores:
            return "unknown"
        # Strategic bombers: a *launch* means cruise missiles, a *takeoff* is aviation.
        if any(b in lowered for b in ("ту-95", "tu-95", "ту-160", "tu-160", "ту-22", "tu-22")):
            if any(w in lowered for w in _LAUNCH_WORDS) and scores["cruise_missile"] == 0:
                scores["cruise_missile"] += 2
        # "уav/дрон" also matches inside "розвідувальний дрон": recon wins when explicit.
        if scores["uav_recon"] and scores["uav_shahed"]:
            scores["uav_shahed"] = 0
        best = max(scores.values())
        for threat in _THREAT_PRIORITY:
            if scores[threat] == best:
                return threat
        return "unknown"

    @staticmethod
    def _find_regions(lowered: str) -> list[_RegionMention]:
        taken = [False] * len(lowered)
        found: list[_RegionMention] = []
        for stem, canonical in _ALIASES_BY_LENGTH:
            start = 0
            while True:
                index = lowered.find(stem, start)
                if index == -1:
                    break
                end = index + len(stem)
                if not any(taken[index:end]):
                    for i in range(index, end):
                        taken[i] = True
                    found.append(
                        _RegionMention(canonical, index, end, canonical in EXTERNAL_ORIGINS)
                    )
                start = end
        found.sort(key=lambda m: m.start)
        # Collapse repeated mentions of the same area, keeping the first.
        seen: set[str] = set()
        unique: list[_RegionMention] = []
        for mention in found:
            if mention.name not in seen:
                seen.add(mention.name)
                unique.append(mention)
        return unique

    @staticmethod
    def _split_source_targets(
        lowered: str, mentions: Sequence[_RegionMention]
    ) -> tuple[Optional[str], list[str]]:
        source: Optional[str] = None
        targets: list[str] = []
        undecided: list[_RegionMention] = []
        for mention in mentions:
            window = lowered[max(0, mention.start - 28): mention.start]
            if mention.external or _SOURCE_MARKER_RE.search(window):
                if source is None:
                    source = mention.name
                elif not mention.external:
                    targets.append(mention.name)
            elif _TARGET_MARKER_RE.search(window):
                targets.append(mention.name)
            else:
                undecided.append(mention)
        has_flow = bool(targets) or bool(re.search(r"курс|напрямк|→|->|towards|heading", lowered))
        for index, mention in enumerate(undecided):
            if source is None and index == 0 and (len(mentions) > 1 and has_flow):
                source = mention.name
            elif mention.name != source and mention.name not in targets:
                targets.append(mention.name)
        targets = [t for t in targets if t in UKRAINIAN_REGIONS]
        return source, targets

    @staticmethod
    def _detect_direction(lowered: str) -> Optional[str]:
        # Only treat compass words as a heading when they describe movement,
        # not when they are part of a region name (e.g. "Південноукраїнськ").
        match = re.search(
            r"(?:курс(?:ом)?|напрям(?:ок|ку)|heading|рухаються|рухається|летять|летить|у бік|в бік)"
            r"[^.;\n]{0,25}?([а-яіїєa-z\-]+)",
            lowered,
        )
        if match:
            direction = _normalise_direction(match.group(0))
            if direction:
                return direction
        for pattern, label in _DIRECTION_PATTERNS[:4]:  # compound headings are unambiguous
            if pattern.search(lowered):
                return label
        return None

    @staticmethod
    def _detect_count(lowered: str) -> Optional[int]:
        match = _COUNT_RE.search(lowered)
        if match:
            value = int(match.group(1))
            return value if 0 < value < 1000 else None
        return None

    @staticmethod
    def _detect_speed(lowered: str) -> Optional[int]:
        match = _SPEED_RE.search(lowered)
        if not match:
            return None
        return int(match.group(1) or match.group(2))

    def _confidence(
        self,
        threat: str,
        source: Optional[str],
        targets: Sequence[str],
        direction: Optional[str],
        count: Optional[int],
        non_empty: bool,
    ) -> float:
        if not non_empty:
            return 0.0
        score = 0.25
        if threat != "unknown":
            score += 0.25
        if source:
            score += 0.12
        if targets:
            score += 0.15 if len(targets) == 1 else 0.18
        if direction:
            score += 0.05
        if count:
            score += 0.05
        return round(min(self.MAX_CONFIDENCE, score), 2)

    @staticmethod
    def _summary(
        threat: str,
        source: Optional[str],
        targets: Sequence[str],
        direction: Optional[str],
        count: Optional[int],
        speed: Optional[int],
    ) -> str:
        parts = [_THREAT_LABELS.get(threat, _THREAT_LABELS["unknown"])]
        if count:
            parts[0] = f"{count}× {parts[0].lower()}"
        route = ""
        if source and targets:
            route = f" from {source} towards {', '.join(targets)}"
        elif targets:
            route = f" over {', '.join(targets)}"
        elif source:
            route = f" from {source}"
        extras = []
        if direction:
            extras.append(f"heading {direction}")
        if speed:
            extras.append(f"~{speed} km/h")
        tail = f" ({'; '.join(extras)})" if extras else ""
        return f"{parts[0]}{route}{tail}."


# --------------------------------------------------------------------------- #
# Synthetic scenarios for offline demos / tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticScenario:
    """A canned message with its expected parse, used in offline mode.

    Attributes:
        name: Stable identifier.
        message: Representative Telegram text.
        keywords: Lower-cased fragments that fingerprint the scenario; all
            must be present for a fuzzy match.
        payload: Field values for :class:`ParsedThreat` (``parser`` is set to
            ``"synthetic"`` automatically).
    """

    name: str
    message: str
    keywords: tuple[str, ...]
    payload: Mapping[str, Any]

    def to_threat(self) -> ParsedThreat:
        """Materialise the scenario as a validated :class:`ParsedThreat`."""
        return ParsedThreat(**{**self.payload, "parser": "synthetic", "source_excerpt": self.message})


SYNTHETIC_SCENARIOS: Final[Sequence[SyntheticScenario]] = (
    SyntheticScenario(
        name="shahed_night_wave_south",
        message="🛵 Група ударних БпЛА (≈12 шахедів) з Херсонщини курсом на Миколаївщину та Одещину, швидкість ~150 км/год",
        keywords=("шахед", "херсон", "миколаїв", "одещин"),
        payload={
            "threat_type": "uav_shahed", "source_region": "Kherson Oblast",
            "target_regions": ["Mykolaiv Oblast", "Odesa Oblast"], "confidence_score": 0.9,
            "raw_summary": "12× Shahed-type strike UAVs from Kherson Oblast towards Mykolaiv and Odesa oblasts (~150 km/h).",
            "direction": "west", "estimated_count": 12,
        },
    ),
    SyntheticScenario(
        name="ballistic_belgorod_kharkiv",
        message="‼️ Пуски балістики з Бєлгородщини на Харків! Негайно в укриття!",
        keywords=("баліст", "бєлгород", "харків"),
        payload={
            "threat_type": "ballistic_missile", "source_region": "Belgorod (RU)",
            "target_regions": ["Kharkiv Oblast"], "confidence_score": 0.95,
            "raw_summary": "Ballistic missile launches from Belgorod towards Kharkiv Oblast; immediate shelter warning.",
            "direction": "south-west", "estimated_count": None,
        },
    ),
    SyntheticScenario(
        name="mig31k_takeoff_savasleyka",
        message="✈️ Зліт МіГ-31К з Саваслейки. Загроза застосування Кинджалів по всій території України.",
        keywords=("міг-31", "зліт", "саваслейк"),
        payload={
            "threat_type": "aviation", "source_region": "Savasleyka (RU)",
            "target_regions": [], "confidence_score": 0.88,
            "raw_summary": "MiG-31K takeoff from Savasleyka; nationwide Kinzhal (ballistic) threat.",
            "direction": None, "estimated_count": 1,
        },
    ),
    SyntheticScenario(
        name="kalibr_black_sea_odesa",
        message="🚀 Пуски Калібрів з акваторії Чорного моря → Одещина, Миколаївщина. 4 ракети.",
        keywords=("калібр", "чорного моря", "одещин"),
        payload={
            "threat_type": "cruise_missile", "source_region": "Black Sea",
            "target_regions": ["Odesa Oblast", "Mykolaiv Oblast"], "confidence_score": 0.92,
            "raw_summary": "4× Kalibr cruise missile launches from the Black Sea towards Odesa and Mykolaiv oblasts.",
            "direction": "north", "estimated_count": 4,
        },
    ),
    SyntheticScenario(
        name="kab_sumy_border",
        message="💥 Тактична авіація скидає КАБи по прикордонню Сумщини (Шостка, Конотоп)",
        keywords=("каб", "сумщин"),
        payload={
            "threat_type": "guided_bomb", "source_region": "Kursk (RU)",
            "target_regions": ["Sumy Oblast"], "confidence_score": 0.85,
            "raw_summary": "Tactical aviation dropping guided bombs on the Sumy Oblast border area (Shostka, Konotop).",
            "direction": None, "estimated_count": None,
        },
    ),
    SyntheticScenario(
        name="recon_uav_dnipro",
        message="👁 Розвідувальний БпЛА типу Орлан над Дніпропетровщиною, рухається на північний захід",
        keywords=("розвід", "орлан", "дніпро"),
        payload={
            "threat_type": "uav_recon", "source_region": None,
            "target_regions": ["Dnipropetrovsk Oblast"], "confidence_score": 0.8,
            "raw_summary": "Orlan reconnaissance UAV over Dnipropetrovsk Oblast heading north-west.",
            "direction": "north-west", "estimated_count": 1,
        },
    ),
)
"""Built-in scenarios returned verbatim in offline mode for matching messages."""


def _normalise_for_match(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text.lower()).split().__str__()


def match_scenario(text: str) -> Optional[SyntheticScenario]:
    """Find the synthetic scenario matching ``text``, if any.

    A scenario matches when the message equals its canonical message
    (ignoring punctuation / emoji / case), or when it contains every
    fingerprint keyword **and** says nothing the canned payload would lose:
    every region the rule parser finds must belong to the scenario, and the
    rule parser's threat type must agree. A pasted report that adds a second
    threat or extra oblasts therefore goes to the rule-based parser instead
    of being replaced by a demo payload.

    Args:
        text: Raw message.

    Returns:
        The matching :class:`SyntheticScenario` or ``None``.
    """
    lowered = (text or "").lower().replace("’", "'")
    if not lowered.strip():
        return None
    normalised = _normalise_for_match(lowered)
    for scenario in SYNTHETIC_SCENARIOS:
        if normalised == _normalise_for_match(scenario.message.lower()):
            return scenario
    rules = RuleBasedOSINTParser()
    mentioned = {mention.name for mention in rules._find_regions(lowered)}
    threat = rules._detect_threat(lowered)
    for scenario in SYNTHETIC_SCENARIOS:
        if not all(keyword in lowered for keyword in scenario.keywords):
            continue
        allowed = set(scenario.payload.get("target_regions") or [])
        if scenario.payload.get("source_region"):
            allowed.add(str(scenario.payload["source_region"]))
        if mentioned <= allowed and threat in (scenario.payload["threat_type"], "unknown"):
            return scenario
    return None


def demo_messages() -> list[str]:
    """Representative raw messages, one per synthetic scenario."""
    return [scenario.message for scenario in SYNTHETIC_SCENARIOS]


# --------------------------------------------------------------------------- #
# Gemini integration
# --------------------------------------------------------------------------- #
class ParserMode(str, Enum):
    """Which engine :class:`GeminiOSINTParser` is currently using."""

    GEMINI = "gemini"
    OFFLINE = "offline"


SYSTEM_PROMPT: Final[str] = f"""You are an OSINT analyst normalising informal Ukrainian air-raid monitoring messages
(Telegram channels; Ukrainian, Russian or English; emoji and slang are common).

Return ONLY a JSON object matching the provided schema. Rules:
1. threat_type must be one of: ballistic_missile, cruise_missile, aviation, guided_bomb,
   artillery, uav_shahed, uav_recon, unknown.
   - Slang: "мопеди", "шахеди", "герані", "ударні БпЛА", "дрони" -> uav_shahed (SLOW wave).
   - "Орлан", "ZALA", "Supercam", "розвідувальний" -> uav_recon (SLOW wave).
   - "балістика", "Іскандер", "Кинджал", "швидкісна ціль", KN-23 -> ballistic_missile (FAST wave).
   - "Калібр", "Х-101/555/22/59", "крилаті", bomber *launches* ("пуски з Ту-95") -> cruise_missile (FAST).
   - "зліт МіГ-31К/Ту-95/Су-34", "тактична авіація" without a launch -> aviation (FAST).
   - "КАБ", "ФАБ з УМПК", "керовані авіабомби" -> guided_bomb (FAST).
   - "обстріл", "РСЗВ", "артилерія" -> artillery (FAST).
2. source_region: where the threat comes from. Use an oblast name from this list or an external
   origin ({", ".join(EXTERNAL_ORIGINS)}). Prepositions "з/із/from" mark the origin. null if absent.
3. target_regions: oblasts projected to be affected ("курсом на", "у напрямку", "→", "по").
   Use canonical English names from: {", ".join(UKRAINIAN_REGIONS)}. Nationwide threats -> [].
4. confidence_score: 0.9+ when type and route are explicit; 0.5-0.7 when inferred; <0.4 if vague.
5. raw_summary: one short English sentence, no emoji, keep counts and speeds if given.
6. direction: compass heading of movement if stated (e.g. "north-west"), else null.
7. estimated_count: integer number of objects if stated, else null.
Never invent regions that are not implied by the text."""


def classify_api_error(exc: BaseException) -> Literal["permanent", "transient"]:
    """Decide whether an exception should disable Gemini for the session.

    Authentication / authorisation problems, a missing SDK and an unknown
    model are *permanent* (retrying would only burn time). Rate limits,
    server errors, timeouts, connection problems and malformed responses are
    *transient* and only count towards the circuit breaker.

    Args:
        exc: The exception raised by the SDK or by response validation.

    Returns:
        ``"permanent"`` or ``"transient"``.
    """
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "permanent"
    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", "") or exc).lower()
    if code in (401, 403, 404):
        return "permanent"
    if code == 400 and ("api key" in message or "api_key" in message or "invalid argument" in message
                        and "model" in message):
        return "permanent"
    if "api key not valid" in message or "api_key_invalid" in message:
        return "permanent"
    return "transient"


def _strip_code_fences(text: str) -> str:
    """Remove ```json fences some models still wrap around JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


class GeminiOSINTParser:
    """Parse OSINT messages with Gemini, falling back to rules automatically.

    The parser never raises from :meth:`parse`. Failure handling:

    * No API key, ``LLM_ENABLED=false`` or SDK not installed -> starts offline.
    * Auth errors (401/403), unknown model (404) or an invalid key -> switches
      to offline for the rest of the session.
    * Timeouts, connection errors, 429/5xx or malformed JSON -> retried once
      with a short backoff; after ``max_consecutive_failures`` failures in a
      row the circuit breaker opens and the parser stays offline.
    * Every failed call still returns a rule-based or synthetic result.

    Args:
        api_key: Gemini API key. Defaults to ``settings.gemini_api_key``.
        model: Model id. Defaults to ``settings.gemini_model``.
        settings: Pre-built :class:`src.config.Settings`; defaults to
            :func:`src.config.get_settings`.
        timeout_s: Per-request HTTP timeout in seconds.
        max_attempts: Attempts per message for transient errors.
        max_consecutive_failures: Failures that open the circuit breaker.
        client_factory: Callable returning an SDK-compatible client (object
            with ``models.generate_content``); used for testing.
        force_offline: Never call the API, even with a valid key.
        use_synthetic_scenarios: Return canned payloads for known demo
            messages when offline.
        sleep: Sleep function used for backoff (injectable for tests).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        settings: Optional[Settings] = None,
        timeout_s: float = 20.0,
        max_attempts: int = 2,
        max_consecutive_failures: int = 3,
        client_factory: Optional[Callable[[str], Any]] = None,
        force_offline: bool = False,
        use_synthetic_scenarios: bool = True,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        resolved = settings or get_settings()
        self.api_key: Optional[str] = (api_key if api_key is not None else resolved.gemini_api_key) or None
        self.model: str = model or resolved.gemini_model
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.use_synthetic_scenarios = use_synthetic_scenarios
        self.rules = RuleBasedOSINTParser()
        self.stats: Counter[str] = Counter()
        self.last_error: Optional[str] = None
        self.offline_reason: Optional[str] = None
        self._client: Any = None
        self._client_factory = client_factory
        self._sleep = sleep
        self._consecutive_failures = 0

        if force_offline:
            self._go_offline("offline mode forced by caller")
        elif not self.api_key:
            self._go_offline("GEMINI_API_KEY is not set")
        elif api_key is None and not resolved.llm_requested:
            self._go_offline("LLM_ENABLED is false")

    # ---- public API ----------------------------------------------------- #
    @property
    def mode(self) -> ParserMode:
        """Current engine."""
        return ParserMode.OFFLINE if self.offline_reason else ParserMode.GEMINI

    @property
    def is_online(self) -> bool:
        """``True`` while Gemini calls are still being attempted."""
        return self.offline_reason is None

    def parse(self, text: str) -> ParsedThreat:
        """Parse a single message, never raising.

        Args:
            text: Raw Telegram message.

        Returns:
            A validated :class:`ParsedThreat`; check ``.parser`` to see which
            engine produced it.
        """
        message = (text or "").strip()
        if not message:
            self.stats["empty"] += 1
            return self.rules.parse("")
        if self.is_online:
            try:
                result = self._parse_with_gemini(message)
            except Exception as exc:  # noqa: BLE001 - every failure must degrade gracefully
                self._record_failure(exc)
            else:
                self._consecutive_failures = 0
                self.stats["gemini"] += 1
                return result
        return self._fallback(message)

    def parse_many(self, texts: Iterable[str]) -> list[ParsedThreat]:
        """Parse several messages sequentially.

        Args:
            texts: Raw messages.

        Returns:
            One :class:`ParsedThreat` per input, in order.
        """
        return [self.parse(text) for text in texts]

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot for dashboards and logs (no secrets)."""
        return {
            "mode": self.mode.value,
            "model": self.model,
            "offline_reason": self.offline_reason,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self.last_error,
            "stats": dict(self.stats),
        }

    # ---- internals ------------------------------------------------------ #
    def _go_offline(self, reason: str) -> None:
        if self.offline_reason is None:
            logger.warning("Gemini parser offline: %s. Using rule-based fallback.", reason)
        self.offline_reason = reason

    def _fallback(self, message: str) -> ParsedThreat:
        if self.use_synthetic_scenarios:
            scenario = match_scenario(message)
            if scenario is not None:
                self.stats["synthetic"] += 1
                return scenario.to_threat()
        self.stats["rules"] += 1
        return self.rules.parse(message)

    def _record_failure(self, exc: BaseException) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"[:300]
        self.stats["errors"] += 1
        kind = classify_api_error(exc)
        if kind == "permanent":
            self._go_offline(self.last_error)
            return
        self._consecutive_failures += 1
        logger.warning("Gemini call failed (%d/%d): %s", self._consecutive_failures,
                       self.max_consecutive_failures, self.last_error)
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._go_offline(
                f"circuit breaker opened after {self._consecutive_failures} consecutive failures "
                f"(last: {self.last_error})"
            )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(self.api_key or "")
            return self._client
        from google import genai  # Imported lazily so the module works without the SDK.
        from google.genai import types

        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )
        return self._client

    def _build_config(self) -> Any:
        """Generation config; a dict when the SDK types are unavailable."""
        base: dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": GeminiThreatSchema,
            "temperature": 0.0,
            "max_output_tokens": 512,
        }
        try:
            from google.genai import types
        except ImportError:
            return base
        if "flash" in self.model.lower():
            # Thinking is unnecessary for extraction and slows the call down.
            base["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**base)

    def _parse_with_gemini(self, message: str) -> ParsedThreat:
        client = self._get_client()
        config = self._build_config()
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=f"Message:\n{message}",
                    config=config,
                )
                return self._to_threat(response, message)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if classify_api_error(exc) == "permanent" or attempt == self.max_attempts:
                    raise
                backoff = 0.5 * attempt
                logger.info("Gemini attempt %d failed (%s); retrying in %.1fs", attempt, exc, backoff)
                self._sleep(backoff)
        assert last_exc is not None  # pragma: no cover - loop always raises or returns
        raise last_exc

    def _to_threat(self, response: Any, message: str) -> ParsedThreat:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        else:
            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise ValueError("Gemini returned an empty response")
            data = json.loads(_strip_code_fences(raw_text))
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise ValueError("Gemini response is not a JSON object")
        try:
            GeminiThreatSchema.model_validate(data)
        except ValidationError as exc:
            logger.info("Gemini payload needed repair: %s", exc.errors()[:2])
        return ParsedThreat(**{**data, "parser": "gemini", "model": self.model,
                               "source_excerpt": message})


if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic + emoji on Windows consoles
    osint = GeminiOSINTParser()
    print("status:", osint.status())
    for sample in demo_messages():
        print("\n>", sample)
        print(osint.parse(sample).model_dump_json(indent=2, exclude={"parsed_at"}))
