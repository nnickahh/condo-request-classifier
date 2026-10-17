from __future__ import annotations

import re

from condo_jev.estate_directory import format_estate_levels_clarification
from condo_jev.schema import Category, Classification, Priority, ResidentRequest

_LEVELS_GUIDE = f" (Provide the block, floor, unit or common-area location.)\n\n{format_estate_levels_clarification()}"


def generate_ai_resident_reply(
    request: ResidentRequest,
    classification: Classification,
) -> str:
    """
    Generate an empathetic, professional concierge-grade AI reply that acknowledges
    the resident's specific situation politely, apologizes for any disturbance or
    inconvenience, and clarifies estate blocks and levels if location is missing.
    """
    text = request.text.strip()
    loc = classification.extracted_location or request.location
    has_loc = bool(loc)
    category = classification.category
    priority = classification.priority
    is_emergency = priority == Priority.EMERGENCY or classification.nouls.get("life_safety_hazard", 0.0) >= 0.50

    # 1. EMERGENCY
    if is_emergency:
        if re.search(r"\b(trapped|stuck|elevator|lift)\b", text, re.IGNORECASE):
            where = f" at {loc}" if has_loc else ""
            if not has_loc:
                return (
                    "EMERGENCY: Lift entrapment report received. Please remain calm inside; mechanical ventilation is active. "
                    "Please provide your lift or block location immediately. Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995`."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"EMERGENCY: Lift entrapment reported{where}. Please remain calm inside; mechanical ventilation is active. "
                "Our emergency response team has been alerted. Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995`."
            )
        if not has_loc:
            return (
                "EMERGENCY: Urgent safety priority. Please provide your unit or block location immediately. "
                "Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995` if immediate danger exists."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"EMERGENCY: Urgent safety priority logged for {loc}. Our on-duty security and emergency team has been alerted. "
            "Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995` if life or safety is in immediate danger."
        )

    # 2. NOISE / NUISANCE
    if category == Category.NOISE:
        if not has_loc:
            return (
                "Noted on the noise disturbance. Please provide your unit or block number so our security team can inspect and enforce estate quiet hours."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Noted on the noise disturbance. I have alerted our 24/7 security patrol regarding {loc} to inspect and enforce estate quiet hours."
        )

    # 3. MAINTENANCE
    elif category == Category.MAINTENANCE:
        # Water / Plumbing leaks
        if re.search(r"\b(leak|dripping|pipe|sink|tap|toilet|flush|water)\b", text, re.IGNORECASE):
            if not has_loc:
                return (
                    "Noted on the water leak. Please provide your unit or block location so our plumbing and mechanical maintenance team can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the water leak. I have logged this request for our plumbing and mechanical maintenance team to inspect at {loc}."
            )
        # Lighting / Electrical
        elif re.search(r"\b(light|flicker|lamp|bulb|dark|switch|power|blackout|mcb|breaker|tripped)\b", text, re.IGNORECASE):
            if not has_loc:
                return (
                    "Noted on the lighting fault. Please provide your unit or block location so our electrical maintenance team can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the lighting fault. I have logged this request for our electrical maintenance team to inspect and rectify at {loc}."
            )
        # Air conditioning
        elif re.search(r"\b(air-?con|warm|cooling|compressor|fan)\b", text, re.IGNORECASE):
            if not has_loc:
                return (
                    "Noted on the air-conditioning issue. Please provide your unit or block location so our HVAC technical team can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the air-conditioning issue. I have logged this request for our HVAC maintenance team to inspect at {loc}."
            )
        # Lift / Elevator
        elif re.search(r"\b(lift|elevator)\b", text, re.IGNORECASE):
            if not has_loc:
                return (
                    "Noted on the elevator issue. Please provide the block or lift location so our lift contractor can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the elevator issue. I have logged this request for our appointed lift maintenance team to inspect at {loc}."
            )
        # Default maintenance
        else:
            if not has_loc:
                return (
                    "Noted on the maintenance issue. Please provide your unit or block location so our facilities team can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the maintenance issue. I have logged this request for our facilities maintenance team to inspect at {loc}."
            )

    # 4. SECURITY
    elif category == Category.SECURITY:
        if not has_loc:
            return (
                "Noted on the security concern. Please provide your unit or block location so our guardhouse team can review and investigate."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Noted on the security concern. I have alerted our 24/7 security patrol regarding {loc} to investigate."
        )

    # 5. PARKING
    elif category == Category.PARKING:
        if not has_loc:
            return (
                "Noted on the parking issue. Please provide your allocated lot number or carpark location (Basement B1/B2) so carpark enforcement can attend."
            )
        return (
            f"Noted on the parking issue. I have logged this report for carpark enforcement to inspect {loc}."
        )

    # 6. CLEANLINESS
    elif category == Category.CLEANLINESS:
        if not has_loc:
            return (
                "Noted on the cleanliness issue. Please provide the location or block so our housekeeping team can attend."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Noted on the cleanliness issue. I have logged this request for our housekeeping and conservancy team to attend to {loc}."
        )

    # 7. FACILITIES
    elif category == Category.FACILITIES:
        if not has_loc:
            return (
                "Thank you for contacting Estate Management. Please provide the facility name or block location so our desk can assist you."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Thank you for contacting Estate Management. I have logged your enquiry regarding {loc} for our facilities desk to follow up."
        )

    # 8. BILLING
    elif category == Category.BILLING:
        if not has_loc:
            return (
                "Thank you for reaching out. Please provide your unit number so our finance office can review your account."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Thank you for reaching out. Our accounts office will review the ledger for unit {loc} and provide an update."
        )

    # 9. RENOVATION
    elif category == Category.RENOVATION:
        if not has_loc:
            return (
                "Thank you for notifying Estate Management. Please provide the unit or block number so we can verify renovation permits."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Thank you for notifying Estate Management. Our management office will verify the renovation permit for {loc}."
        )

    # 10. GENERAL / OTHER
    else:
        if not has_loc:
            return (
                "Thank you for contacting Estate Management. Please provide your unit or block location so we can assist you."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"Thank you for contacting Estate Management. I have logged your enquiry regarding {loc} for our estate office to follow up."
        )
