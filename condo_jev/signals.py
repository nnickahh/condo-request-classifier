import re
from dataclasses import dataclass
from typing import Literal

from condo_jev.schema import Category, Priority

SignalKind = Literal["emergency_hazard", "prompt_injection", "sensitive_escalation"]


@dataclass(frozen=True)
class SafetyScreenSignal:
    kind: SignalKind
    category: Category
    priority: Priority
    reason: str
    matched_span: str


_EMERGENCY_PATTERNS: list[tuple[re.Pattern[str], Category, str]] = [
    (
        re.compile(
            r"\b(trapped\s+in\s+(the\s+)?(lift|elevator)|stuck\s+inside\s+(the\s+)?(lift|elevator)|lift\s+stuck\s+with\s+(people|child|passenger|someone))\b",
            re.IGNORECASE,
        ),
        Category.MAINTENANCE,
        "Life-safety emergency: passenger reported trapped inside lift.",
    ),
    (
        re.compile(
            r"\b(smell\s+of\s+gas|gas\s+leak|smoke\s+coming\s+from|fire\s+alarm\s+ringing|flames\s+in|burning\s+smell\s+from\s+(switchboard|panel|riser)|sparks\s+flying|electrical\s+fire)\b",
            re.IGNORECASE,
        ),
        Category.MAINTENANCE,
        "Life-safety emergency: fire, smoke, gas, or acute electrical hazard detected.",
    ),
    (
        re.compile(
            r"\b(water\s+gushing|burst\s+(water\s+)?(pipe|main)|flooding\s+the\s+(corridor|lobby|lift|unit|basement)|ceiling\s+collapsed)\b",
            re.IGNORECASE,
        ),
        Category.MAINTENANCE,
        "Critical infrastructure emergency: active flooding, burst main, or structural collapse.",
    ),
    (
        re.compile(
            r"\b(breaking\s+into|active\s+break-?in|intruder\s+with|violent\s+fight|someone\s+(is\s+)?trying\s+to\s+force\s+open|armed|physical\s+assault)\b",
            re.IGNORECASE,
        ),
        Category.SECURITY,
        "Immediate security emergency: active break-in or physical threat reported.",
    ),
]

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules)\b", re.IGNORECASE),
    re.compile(r"\b(system\s+override|developer\s+mode|reveal\s+(your\s+)?system\s+prompt)\b", re.IGNORECASE),
    re.compile(r"\b(output\s+only\s+json|set\s+category\s+to|do\s+not\s+flag\s+for\s+review|bypass\s+review)\b", re.IGNORECASE),
]

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(police\s+report|my\s+lawyer|legal\s+action|suing\s+the\s+mcst|pdpa\s+(breach|complaint)|harassment|stalking|theft\s+of)\b", re.IGNORECASE),
]

_HIGH_PRIORITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(urgent|urgently|asap|immediately|right\s+now|locked\s+out|no\s+water|power\s+trip|tripping\s+the\s+circuit|jerky\s+stopping|unresponsive\s+door|blackout|blocking\s+my\s+(car|lot)|hornet|wasp|snake|rat\s+inside)\b",
        re.IGNORECASE,
    ),
]

_LOW_PRIORITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(just\s+wondering|whenever\s+convenient|no\s+rush|next\s+month|general\s+question|office\s+hours|agm\s+minutes|by-?laws?\s+copy|feedback\s+on)\b",
        re.IGNORECASE,
    ),
]

_LOCATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"#\d{1,2}-\d{1,4}[A-Za-z]?\b"),
    re.compile(r"\b(?:Blk|Block|Tower)\s*[A-Za-z0-9]{1,4}(?:\s*Level\s*\d{1,2})?\b", re.IGNORECASE),
    re.compile(r"\b(?:Basement|Level|Floor)\s*[B0-9]{1,3}\b", re.IGNORECASE),
    re.compile(r"\b(?:Lift|Elevator)\s*(?:Lobby\s*)?[A-F1-9]\b", re.IGNORECASE),
    re.compile(r"\b(?:BBQ\s*Pit\s*\d?|Function\s*Room\s*[A-Z0-9]?|Tennis\s*Court\s*\d?|Clubhouse|Guardhouse|Main\s*Gate|Side\s*Gate|Bin\s*Centre|Swimming\s*Pool|Gym)\b", re.IGNORECASE),
]

_TOPIC_KEYWORD_GROUPS: dict[Category, tuple[str, ...]] = {
    Category.MAINTENANCE: ("leak", "pipe", "light", "bulb", "lift", "aircon", "air-con", "broken", "faulty", "repair", "water heater", "intercom", "pump", "flickering"),
    Category.SECURITY: ("access card", "key fob", "guard", "cctv", "tailgat", "suspicious", "stranger", "side gate", "intercom access", "trespass", "security"),
    Category.NOISE: ("loud", "music", "party", "barking", "dog", "noise", "midnight", "drilling", "stomping", "karaoke", "quiet hours", "smoke smell"),
    Category.CLEANLINESS: ("rubbish", "chute", "bin", "cockroach", "pest", "mosquito", "dirty", "spill", "smelly", "garbage", "rat", "ant", "cleaning"),
    Category.FACILITIES: ("book", "reservation", "reserve", "bbq", "function room", "tennis court", "badminton", "gym slot", "clubhouse", "karaoke room"),
    Category.PARKING: ("parking", "carpark", "lot", "iu", "transponder", "wheel clamp", "visitor pass", "barrier", "ev charger", "license plate", "parked in my"),
    Category.BILLING: ("invoice", "giro", "levy", "maintenance fee", "deposit refund", "receipt", "charged twice", "sinking fund", "payment", "statement", "billing"),
    Category.RENOVATION: ("renovation", "contractor", "hacking", "moving", "mover", "bulky", "delivery", "lift padding", "permit", "work permit"),
    Category.GENERAL: ("office hours", "agm", "minutes", "by-law", "bylaw", "parcel", "notice", "circular", "management office", "contact number", "policy"),
}

_MULTI_CONNECTORS = re.compile(
    r"\b(also|separately|another\s+thing|by\s+the\s+way|on\s+a\s+separate\s+note|secondly|in\s+addition|plus\s+can\s+you)\b",
    re.IGNORECASE,
)


def screen_request_safety(text: str) -> SafetyScreenSignal | None:
    """Deterministic safety & adversarial screen executed before model inference."""
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return SafetyScreenSignal(
                kind="prompt_injection",
                category=Category.OTHER,
                priority=Priority.HIGH,
                reason="Prompt-injection or policy-override instruction detected in resident input.",
                matched_span=match.group(0),
            )

    for pattern, category, reason in _EMERGENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            return SafetyScreenSignal(
                kind="emergency_hazard",
                category=category,
                priority=Priority.EMERGENCY,
                reason=reason,
                matched_span=match.group(0),
            )

    for pattern in _SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            return SafetyScreenSignal(
                kind="sensitive_escalation",
                category=Category.SECURITY,
                priority=Priority.HIGH,
                reason="Sensitive legal, police, privacy, or harassment term requires officer escalation.",
                matched_span=match.group(0),
            )

    return None


def extract_location(text: str, explicit_location: str | None = None) -> str | None:
    """Return explicit location if provided, or extract unit/block/facility mention from text."""
    if explicit_location and explicit_location.strip():
        return explicit_location.strip()
    found: list[str] = []
    for pattern in _LOCATION_PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(0).strip()
            if span not in found:
                found.append(span)
    if not found:
        return None
    return ", ".join(found[:2])


def detect_matched_topic_groups(text: str) -> list[Category]:
    """Return all distinct domain categories whose indicator keywords appear in `text`."""
    lowered = text.lower()
    matched: list[Category] = []
    for category, keywords in _TOPIC_KEYWORD_GROUPS.items():
        if any(kw in lowered for kw in keywords):
            matched.append(category)
    return matched


def compute_noul_probabilities(
    text: str,
    choice_confidence: float,
    chosen_category: Category,
) -> dict[str, float]:
    """Compute calibrated [0, 1] Noul probabilities for the 4 binary Jev questions."""
    signal = screen_request_safety(text)

    # 1. life_safety_hazard
    if signal and signal.kind == "emergency_hazard":
        noul_hazard = 0.97
    elif any(p.search(text) for p in _HIGH_PRIORITY_PATTERNS) and chosen_category in {Category.MAINTENANCE, Category.SECURITY}:
        noul_hazard = 0.28
    else:
        noul_hazard = 0.03

    # 2. multi_intent_request
    matched_topics = detect_matched_topic_groups(text)
    has_connector = bool(_MULTI_CONNECTORS.search(text))
    if len(matched_topics) >= 3 and has_connector:
        noul_multi = 0.91
    elif len(matched_topics) >= 2 and has_connector:
        noul_multi = 0.84
    elif len(matched_topics) >= 3:
        noul_multi = 0.58
    elif len(matched_topics) == 2:
        noul_multi = 0.24
    elif has_connector and len(text.split()) > 18:
        noul_multi = 0.38
    else:
        noul_multi = 0.05

    # 3. adversarial_or_sensitive
    if signal and signal.kind == "prompt_injection":
        noul_adv = 0.98
    elif signal and signal.kind == "sensitive_escalation":
        noul_adv = 0.91
    else:
        noul_adv = 0.02

    # 4. requires_human_triage
    if choice_confidence < 0.55:
        uncertainty_risk = min(0.95, 0.52 + (0.55 - choice_confidence) * 1.1)
    else:
        uncertainty_risk = max(0.02, (1.0 - choice_confidence) * 0.70)

    other_penalty = 0.88 if chosen_category == Category.OTHER else 0.0
    short_penalty = 0.75 if len(text.strip().split()) < 4 else 0.0
    combined_review_prob = max(
        noul_hazard,
        noul_adv,
        noul_multi if noul_multi >= 0.50 else 0.0,
        other_penalty,
        short_penalty,
        uncertainty_risk,
    )
    noul_review = round(min(0.99, max(0.02, combined_review_prob)), 4)

    return {
        "life_safety_hazard": round(noul_hazard, 4),
        "multi_intent_request": round(noul_multi, 4),
        "adversarial_or_sensitive": round(noul_adv, 4),
        "requires_human_triage": noul_review,
    }


def compute_urgency_score_distribution(
    text: str,
    chosen_category: Category,
) -> tuple[float, Priority, float, dict[str, float]]:
    """
    Compute a Jev `Score` primitive output:
    (continuous_score_0_to_1, Priority_level, confidence, level_probabilities_dict).
    """
    signal = screen_request_safety(text)
    if signal and signal.kind == "emergency_hazard":
        probs = {
            Priority.LOW.value: 0.01,
            Priority.NORMAL.value: 0.02,
            Priority.HIGH.value: 0.09,
            Priority.EMERGENCY.value: 0.88,
        }
        return 0.95, Priority.EMERGENCY, 0.89, probs

    if (signal and signal.kind in {"prompt_injection", "sensitive_escalation"}) or any(
        p.search(text) for p in _HIGH_PRIORITY_PATTERNS
    ):
        probs = {
            Priority.LOW.value: 0.03,
            Priority.NORMAL.value: 0.16,
            Priority.HIGH.value: 0.72,
            Priority.EMERGENCY.value: 0.09,
        }
        return 0.72, Priority.HIGH, 0.74, probs

    if any(p.search(text) for p in _LOW_PRIORITY_PATTERNS) or chosen_category == Category.GENERAL:
        probs = {
            Priority.LOW.value: 0.74,
            Priority.NORMAL.value: 0.22,
            Priority.HIGH.value: 0.03,
            Priority.EMERGENCY.value: 0.01,
        }
        return 0.18, Priority.LOW, 0.76, probs

    probs = {
        Priority.LOW.value: 0.14,
        Priority.NORMAL.value: 0.73,
        Priority.HIGH.value: 0.11,
        Priority.EMERGENCY.value: 0.02,
    }
    return 0.44, Priority.NORMAL, 0.74, probs


def extract_evidence_spans(text: str, max_spans: int = 3) -> list[str]:
    """Extract verbatim substrings from `text` that triggered domain/safety signals."""
    spans: list[str] = []
    signal = screen_request_safety(text)
    if signal and signal.matched_span in text:
        spans.append(signal.matched_span)

    loc = extract_location(text)
    if loc:
        for part in loc.split(", "):
            if part in text and part not in spans:
                spans.append(part)

    for keywords in _TOPIC_KEYWORD_GROUPS.values():
        for kw in keywords:
            match = re.search(rf"\b\w*{re.escape(kw)}\w*\b", text, flags=re.IGNORECASE)
            if match:
                verbatim = match.group(0)
                if verbatim in text and verbatim not in spans:
                    spans.append(verbatim)
                if len(spans) >= max_spans:
                    return spans[:max_spans]
    return spans[:max_spans]
