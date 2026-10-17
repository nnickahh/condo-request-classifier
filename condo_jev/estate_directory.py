"""
Estate directory and block/level validation rules for Condominium Management.
Provides estate layout definitions, level ranges, and validation logic.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Estate layout definition
# Block A: Levels 1 to 24 (Basements: B1, B2)
# Block B: Levels 1 to 24 (Basements: B1, B2)
# Block C: Levels 1 to 20 (Basements: B1, B2)
ESTATE_BLOCKS: dict[str, int] = {
    "A": 24,
    "B": 24,
    "C": 20,
}

MAX_ESTATE_LEVEL = 24
MIN_ESTATE_LEVEL = 1

VALID_BASEMENTS = {"B1", "B2"}

COMMON_FACILITIES = (
    "gym",
    "swimming pool",
    "pool",
    "clubhouse",
    "tennis court",
    "bbq",
    "bbq pit",
    "bbq pits",
    "function room",
    "guardhouse",
    "main gate",
    "side gate",
    "bin centre",
    "bin center",
    "carpark",
    "car park",
    "playground",
    "fitness corner",
)

ESTATE_LEVELS_CLARIFICATION = (
    "Estate Blocks & Levels:\n"
    "• Block A: Levels 1 to 24\n"
    "• Block B: Levels 1 to 24\n"
    "• Block C: Levels 1 to 20"
)


class ValidationResult(NamedTuple):
    is_valid: bool
    parsed_location: str | None
    error_message: str | None


def format_estate_levels_clarification() -> str:
    """Return standard estate block and level guidance for residents."""
    return ESTATE_LEVELS_CLARIFICATION


def validate_estate_location(text: str) -> ValidationResult:
    """
    Validate whether a resident-provided location or level is valid within this estate.

    Returns:
        ValidationResult(
            is_valid=True/False,
            parsed_location=extracted location string or None,
            error_message=explanation when level/block is invalid, else None
        )
    """
    clean = text.strip()
    if not clean:
        return ValidationResult(False, None, "No location was provided.")

    lowered = clean.lower()

    # 1. Extract block if mentioned: "Block A", "Blk B", "Tower C", etc.
    block_match = re.search(r"\b(?:block|blk|tower)\s*([A-Za-z0-9]{1,4})\b", clean, re.IGNORECASE)
    block_letter: str | None = None
    if block_match:
        cand = block_match.group(1).lower()
        non_block_words = {"but", "and", "or", "is", "was", "the", "not", "with", "at", "for", "in", "on", "to", "of", "from", "i", "we", "he", "she", "they"}
        if cand not in non_block_words:
            raw_block = cand.upper()
            if raw_block not in ESTATE_BLOCKS:
                return ValidationResult(
                    False,
                    None,
                    f"Block {raw_block} is not a valid block in this estate.\n\n"
                    f"{ESTATE_LEVELS_CLARIFICATION}\n\n"
                    f"Please verify and resend again.",
                )
            block_letter = raw_block

    # 2. Extract basement level if explicitly specified: "Basement 3", "B3", "Level B1"
    b_explicit = re.search(r"\b(?:basement\s*|level\s+b|lvl\s+b)([0-9]{1,2})\b", clean, re.IGNORECASE)
    if b_explicit:
        b_num = f"B{b_explicit.group(1)}"
        if b_num not in VALID_BASEMENTS:
            return ValidationResult(
                False,
                None,
                f"This is not a valid level. Basement {b_num} does not exist in this estate "
                f"(valid basement levels are B1 and B2).\n\n"
                f"Please verify and resend again.",
            )
        return ValidationResult(True, clean, None)

    # 3. Extract unit notation (#08-14, #B1-02, unit 08-14, 08-14)
    level_num: int | None = None
    unit_m = re.search(r"#([B0-9]{1,2})[-\s](\d{1,4})[A-Za-z]?\b", clean, re.IGNORECASE)
    if not unit_m:
        unit_m = re.search(r"\b(?:unit|apt|apartment)\s*#?([B0-9]{1,2})[-\s](\d{1,4})[A-Za-z]?\b", clean, re.IGNORECASE)
    if not unit_m:
        unit_m = re.search(r"(?<![+\d])\b([0-9]{1,2})-(\d{1,4})[A-Za-z]?\b", clean)

    if unit_m:
        lvl_part = unit_m.group(1).upper()
        if lvl_part.startswith("B"):
            if lvl_part not in VALID_BASEMENTS:
                return ValidationResult(
                    False,
                    None,
                    f"This is not a valid level. Basement {lvl_part} does not exist in this estate "
                    f"(valid basement levels are B1 and B2).\n\n"
                    f"Please verify and resend again.",
                )
            return ValidationResult(True, clean, None)
        else:
            try:
                level_num = int(lvl_part)
            except ValueError:
                level_num = None

    # 4. Extract explicit level notation if unit pattern was not found:
    # "Level 28", "lvl 22", "Floor 12", "28th floor", "Level 8"
    if level_num is None:
        lvl_match = re.search(r"\b(?:level|lvl|floor)\s*(\d{1,2})\b", clean, re.IGNORECASE)
        if lvl_match:
            level_num = int(lvl_match.group(1))
        else:
            ord_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s*(?:level|floor)\b", clean, re.IGNORECASE)
            if ord_match:
                level_num = int(ord_match.group(1))

    # 5. Validate extracted level
    if level_num is not None:
        # Check level 0
        if level_num < MIN_ESTATE_LEVEL:
            return ValidationResult(
                False,
                None,
                f"This is not a valid level. Residential levels start at Level 1.\n\n"
                f"{ESTATE_LEVELS_CLARIFICATION}\n\n"
                f"Please verify and resend again.",
            )

        # If a block was identified, check against that block's maximum level
        if block_letter:
            max_block_lvl = ESTATE_BLOCKS[block_letter]
            if level_num > max_block_lvl:
                return ValidationResult(
                    False,
                    None,
                    f"This is not a valid level. Block {block_letter} only has levels 1 to {max_block_lvl}.\n\n"
                    f"{ESTATE_LEVELS_CLARIFICATION}\n\n"
                    f"Please verify and resend again.",
                )
        else:
            # If no block was specified, check against estate maximum
            if level_num > MAX_ESTATE_LEVEL:
                return ValidationResult(
                    False,
                    None,
                    f"This is not a valid level. Level {level_num} does not exist in this estate "
                    f"(valid levels are 1 to {MAX_ESTATE_LEVEL}).\n\n"
                    f"{ESTATE_LEVELS_CLARIFICATION}\n\n"
                    f"Please verify and resend again.",
                )

    # 6. Check for common facilities
    for fac in COMMON_FACILITIES:
        if fac in lowered:
            return ValidationResult(True, clean, None)

    # 7. Check if text specifies a valid block without level (e.g. "Block B")
    if block_letter and level_num is None:
        return ValidationResult(True, f"Block {block_letter}", None)

    # 8. Check carpark lots (e.g. "Lot B2-15", "Lot 12")
    if re.search(r"\b(?:lot|carpark|bay)\s*[A-Za-z0-9\-]+\b", clean, re.IGNORECASE):
        return ValidationResult(True, clean, None)

    # If a valid level or unit was found
    if level_num is not None:
        return ValidationResult(True, clean, None)

    return ValidationResult(True, clean, None)


class ParsedEstateComponents(NamedTuple):
    block: str | None
    level: str | None
    unit: str | None
    reason: str | None
    formatted_location: str | None


def extract_estate_components(text: str, loc_hint: str | None = None) -> ParsedEstateComponents:
    """
    Extract structured estate parameters (block, level, unit, reason) from resident report and location text.
    Handles multiple conversational styles:
    - "block a, level 8, unit #08-14 and the reason is corridor light flickering"
    - "it's at block a"
    - "#14-08"
    - "block b level 14 unit 12 water leak in kitchen"
    """
    combined = f"{loc_hint or ''} {text or ''}".strip()
    block: str | None = None
    level: str | None = None
    unit: str | None = None
    reason: str | None = None

    # 1. Block: Block A, Blk B, Tower C
    bm = re.search(r"\b(?:block|blk|tower)\s*([A-Za-z0-9]{1,4})\b", combined, re.IGNORECASE)
    if bm:
        cand_b = bm.group(1).lower()
        if cand_b not in {"but", "and", "or", "is", "was", "the", "not", "with", "at", "for", "in", "on", "to", "of", "from", "i", "we", "he", "she", "they"}:
            raw_b = cand_b.upper()
            block = f"Block {raw_b}"

    # 2. Level explicit (e.g. "level 8", "14th floor", "basement 1", "level b1")
    lm = re.search(r"\b(?:level|lvl|floor)\s*([B0-9]{1,2})\b", combined, re.IGNORECASE)
    if not lm:
        lm = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s*(?:level|floor)\b", combined, re.IGNORECASE)
    if not lm:
        lm = re.search(r"\b(?:basement\s*|level\s+b|lvl\s+b)([0-9]{1,2})\b", combined, re.IGNORECASE)
        if lm:
            level = f"Level B{lm.group(1)}"
    if lm and not level:
        raw_l = lm.group(1).upper()
        level = f"Level {raw_l}" if raw_l.startswith("B") else f"Level {int(raw_l)}"

    # 3. Unit standard formats: #08-14, 08-14, unit #08-14, #B1-02
    um = re.search(r"#([B0-9]{1,2})[-\s](\d{1,4})[A-Za-z]?\b", combined, re.IGNORECASE)
    if not um:
        um = re.search(r"\b(?:unit|apt|apartment)\s*#?([B0-9]{1,2})[-\s](\d{1,4})[A-Za-z]?\b", combined, re.IGNORECASE)
    if not um:
        um = re.search(r"(?<![+\d])\b([0-9]{1,2})-(\d{1,4})[A-Za-z]?\b", combined)
    if not um:
        um = re.search(r"\b(?:outside|unit|at|in)\s*#?(\d{2})(\d{2})\b", combined, re.IGNORECASE)

    if um:
        fl = um.group(1).upper()
        un = um.group(2)
        unit = f"#{fl.zfill(2) if fl.isdigit() else fl}-{un.zfill(2)}"
        if not level:
            if fl.startswith("B"):
                level = f"Level {fl}"
            elif fl.isdigit():
                level = f"Level {int(fl)}"
    else:
        # Check for simple unit notation e.g. "unit 14" when level is known
        simple_u = re.search(r"\b(?:unit|apt|door)\s*#?(\d{1,4})[A-Za-z]?\b", combined, re.IGNORECASE)
        if simple_u:
            raw_u = simple_u.group(1)
            if level and level.startswith("Level ") and level.replace("Level ", "").isdigit():
                lvl_val = int(level.replace("Level ", ""))
                unit = f"#{lvl_val:02d}-{int(raw_u):02d}"
            else:
                unit = f"#{raw_u}"

    # 4. Reason / Issue
    rm = re.search(r"\b(?:reason|issue|problem|fault)(?:\s+is|\s*:)?\s*(.+)$", text, re.IGNORECASE)
    if rm:
        reason = rm.group(1).strip()
    else:
        cleaned_no_loc = text.strip()
        if block:
            cleaned_no_loc = re.sub(r"\b(?:it\'s at|its at|at|in)?\s*(?:block|blk|tower)\s*[A-Za-z0-9]+\b", "", cleaned_no_loc, flags=re.I)
        if unit:
            cleaned_no_loc = re.sub(r"\b(?:unit|apt|door)?\s*#?[B0-9]{1,2}[-\s]\d{1,4}[A-Za-z]?\b", "", cleaned_no_loc, flags=re.I)
            cleaned_no_loc = re.sub(r"\b(?:outside|unit|at|in)\s*#?\d{2,4}\b", "", cleaned_no_loc, flags=re.I)
        if level:
            cleaned_no_loc = re.sub(r"\b(?:level|lvl|floor)\s*[B0-9]{1,2}\b", "", cleaned_no_loc, flags=re.I)
            cleaned_no_loc = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\s*(?:level|floor)\b", "", cleaned_no_loc, flags=re.I)
        cleaned_no_loc = re.sub(r"^\s*(?:it\'s at|its at|at|in|and|the)\b\s*", "", cleaned_no_loc, flags=re.I)
        cleaned_no_loc = re.sub(r"^[,;.\s]+|[,;.\s]+$", "", cleaned_no_loc).strip()
        if len(cleaned_no_loc) >= 4 and not re.match(r"^#?[0-9\-]+$", cleaned_no_loc):
            reason = cleaned_no_loc
        else:
            reason = text.strip()

    # Build formatted location string
    loc_parts: list[str] = []
    if block:
        loc_parts.append(block)
    if unit:
        loc_parts.append(unit)
    elif level:
        loc_parts.append(level)
    formatted_loc = ", ".join(loc_parts) if loc_parts else (loc_hint or None)

    return ParsedEstateComponents(
        block=block,
        level=level,
        unit=unit,
        reason=reason,
        formatted_location=formatted_loc,
    )


def is_concrete_estate_location(location: str | None, text: str = "") -> bool:
    """
    Determine if a report or location string specifies a concrete estate location.
    Returns True if a block, level, unit, lot, or recognized estate facility is identified.
    Returns False if only a generic issue description is present (e.g. "there's a water leak").
    """
    if location and location.strip():
        loc_clean = location.strip()
        if loc_clean.lower() in ("pending confirmation", "pending location confirmation", "unspecified", "none"):
            pass
        else:
            comp_loc = extract_estate_components(loc_clean)
            if comp_loc.unit or comp_loc.block or comp_loc.level:
                return True
            lowered_loc = loc_clean.lower()
            if any(fac in lowered_loc for fac in COMMON_FACILITIES):
                return True
            if re.search(r"\b(lot|bay|basement|b[12]|lift|lobby|gate)\b", lowered_loc):
                return True

    if text and text.strip():
        comp_text = extract_estate_components(text)
        if comp_text.unit or comp_text.block or comp_text.level:
            return True
        lowered_text = text.lower()
        if any(fac in lowered_text for fac in COMMON_FACILITIES):
            return True
        if re.search(r"\b(lot\s+[A-Za-z0-9\-]+|basement\s*[12]|b[12]|lift\s+[A-Za-z0-9]|lobby\s+[A-Za-z0-9])\b", lowered_text):
            return True

    return False


class ParsedNoiseDetails(NamedTuple):
    is_noise: bool
    start_time: str | None
    duration: str | None
    noise_type: str | None
    has_timing_details: bool


def is_noise_complaint(text: str) -> bool:
    """Detect if the resident's message is reporting a noise disturbance."""
    if not text:
        return False
    lower = text.lower()
    noise_pattern = re.compile(
        r"\b("
        r"noise|noisy|loud\s+(?:music|sound|bass|banging|drilling|thumping|party|partying|shouting|singing|karaoke|screaming|hammering)|"
        r"banging\s+(?:sound|noise|doors?|walls?|ceiling|floor|at\s+night)?|"
        r"drilling\s+(?:sound|noise)?|"
        r"dog\s+barking|barking\s+dog|"
        r"screaming|stomping|renovation\s+noise|midnight\s+noise|late\s+night\s+noise|"
        r"sound\s+at\s+night|noise\s+at\s+night|loud\s+at\s+night|"
        r"neighbour\s+(?:making\s+noise|party|loud|shouting)|neighbor\s+(?:making\s+noise|party|loud|shouting)"
        r")\b",
        re.IGNORECASE,
    )
    return bool(noise_pattern.search(lower))


def extract_noise_details(text: str) -> ParsedNoiseDetails:
    """
    Extract noise timing details: what time the sound started and how long it has been going on.
    """
    if not is_noise_complaint(text):
        return ParsedNoiseDetails(
            is_noise=False,
            start_time=None,
            duration=None,
            noise_type=None,
            has_timing_details=False,
        )

    lower = text.lower()
    start_time: str | None = None
    duration: str | None = None
    noise_type: str | None = None

    # Detect specific noise type
    type_matches = [
        ("loud music", r"\b(?:loud\s+music|music|bass|karaoke)\b"),
        ("drilling / hammering", r"\b(?:drilling|hammering|renovation)\b"),
        ("dog barking", r"\b(?:dog\s+barking|barking)\b"),
        ("banging / stomping", r"\b(?:banging|stomping|thumping|jumping)\b"),
        ("shouting / partying", r"\b(?:shouting|party|partying|screaming|yelling)\b"),
    ]
    for n_label, n_pat in type_matches:
        if re.search(n_pat, lower, re.IGNORECASE):
            noise_type = n_label
            break

    # 1. Start time extraction
    # "started at 11pm", "since 10:30pm", "started around midnight", "began at 1am", "at 11pm", "around 10pm"
    st_match = re.search(
        r"\b(?:start(?:ed|ing)?|began|since|commenced|from)\s+(?:around\s+|at\s+|about\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|midnight|noon)\b",
        lower,
    )
    if not st_match:
        st_match = re.search(r"\b(?:at|around)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower)
    if not st_match:
        st_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower)
    if not st_match:
        st_match = re.search(r"\b(midnight|past\s+midnight)\b", lower)

    if st_match:
        raw_st = st_match.group(1).strip()
        raw_st = re.sub(r"(\d)(am|pm)", r"\1 \2", raw_st, flags=re.I)
        start_time = raw_st.title() if ("midnight" in raw_st.lower() or "noon" in raw_st.lower()) else raw_st.upper()

    # 2. Duration extraction
    # "for 2 hours", "past 45 minutes", "going on for an hour", "non-stop for 30 mins", "all night"
    dur_match = re.search(
        r"\b(?:going\s+on\s+for|for|past|around|about|last)\s+(\d+\s*(?:hours?|hrs?|mins?|minutes?))\b",
        lower,
    )
    if not dur_match:
        dur_match = re.search(
            r"\b(?:for\s+)?(?:an\s+hour|half\s+an\s+hour|a\s+couple\s+of\s+hours|a\s+few\s+hours|all\s+night|whole\s+night|non-?stop)\b",
            lower,
        )
        if dur_match:
            raw_d = dur_match.group(0).strip()
            if "half" in raw_d:
                duration = "30 minutes"
            elif "an hour" in raw_d:
                duration = "1 hour"
            elif "couple" in raw_d:
                duration = "2 hours"
            elif "all night" in raw_d or "whole night" in raw_d:
                duration = "All night"
            elif "non" in raw_d:
                duration = "Non-stop"
    else:
        duration = dur_match.group(1).strip()

    # Both start_time and duration are present
    has_timing = bool(start_time and duration)

    return ParsedNoiseDetails(
        is_noise=True,
        start_time=start_time,
        duration=duration,
        noise_type=noise_type,
        has_timing_details=has_timing,
    )


def format_noise_clarification_prompt(location_str: str | None = None) -> str:
    """Format professional clarification asking for noise duration and start time without emojis."""
    loc_ack = f"We have noted your location as `{location_str}`.\n\n" if location_str else ""
    return (
        f"*NOISE DISTURBANCE REPORT*\n"
        f"──────────────────────────────\n"
        f"Thank you for bringing this noise disturbance to our attention.\n\n"
        f"{loc_ack}"
        f"To assist our on-duty security team in investigating and taking prompt action, could you please provide a few details:\n\n"
        f"• *What time did the sound start?* (e.g. _'around 11:00 PM'_)\n"
        f"• *How long has it been going on?* (e.g. _'for about an hour'_)\n"
        f"• *Which block, level, or unit is the noise coming from?* (if known)\n\n"
        f"Please reply with these details so our guardhouse can conduct an inspection."
    )

