from __future__ import annotations

import os
import re
import sys

from condo_jev.estate_directory import format_estate_levels_clarification
from condo_jev.schema import Category, Classification, Priority, ResidentRequest

_LEVELS_GUIDE = f" (Provide the block, floor, unit or common-area location.)\n\n{format_estate_levels_clarification()}"


def _generate_gemini_flash_resident_reply(
    request: ResidentRequest,
    classification: Classification,
    has_loc: bool,
    loc: str | None,
) -> str | None:
    """
    Generate a concise, empathetic concierge reply using Gemini Flash (`gemini-2.0-flash`).
    Automatically bypassed during unit tests or if GEMINI_API_KEY is not set.
    """
    if (
        os.getenv("PYTEST_CURRENT_TEST")
        or "pytest" in sys.modules
        or os.getenv("CONDO_TEST_MODE") == "1"
        or os.getenv("CONDO_DISABLE_LLM_REPLIES") == "1"
    ) and not os.getenv("CONDO_FORCE_LIVE_GEMINI"):
        return None

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    loc_instruction = (
        f"The location is {loc}. Confirm that you have logged the issue for {classification.assigned_team} to attend to {loc}."
        if has_loc
        else f"The location is missing. Politely ask the resident to provide their unit or block location so {classification.assigned_team} can attend."
    )

    prompt = (
        "You are the Estate Management Concierge for Emerald Bay Residences (MCST 4521). "
        "Write a polite, concise, 1-to-2 sentence acknowledgement reply to the resident's report. "
        "Do NOT include greetings like 'Dear Resident' or sign-offs like 'Best regards'. "
        "Do NOT use emojis. Address the specific issue reported and mention the assigned team.\n\n"
        f"Resident message: {request.text.strip()}\n"
        f"Category: {classification.category_title}\n"
        f"Assigned Team: {classification.assigned_team}\n"
        f"Priority: {classification.priority.value.upper()}\n"
        f"Instruction: {loc_instruction}"
    )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        model_name = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
        for m in [model_name, "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                res = client.models.generate_content(model=m, contents=prompt)
                if res.text and len(res.text.strip()) > 15:
                    content = res.text.strip()
                    if not has_loc:
                        return f"{content}{_LEVELS_GUIDE}"
                    return content
            except Exception:
                continue
    except Exception:
        pass
    return None


def generate_ai_resident_reply(
    request: ResidentRequest,
    classification: Classification,
) -> str:
    """
    Generate an empathetic, professional concierge-grade AI reply that acknowledges
    the resident's specific situation politely, apologizes for any disturbance or
    inconvenience, and clarifies estate blocks and levels if location is missing.
    Uses Gemini Flash when GEMINI_API_KEY is available, with instant deterministic fallback.
    """
    text = request.text.strip()
    loc = classification.extracted_location or request.location
    has_loc = bool(loc)
    category = classification.category
    priority = classification.priority
    is_emergency = priority == Priority.EMERGENCY or classification.nouls.get("life_safety_hazard", 0.0) >= 0.50

    # 1. EMERGENCY (Always immediate deterministic life-safety instructions & human agent handover)
    if is_emergency:
        # Child / infant entrapment, head stuck, severe fall, medical crisis
        if re.search(r"\b(baby|infant|child|toddler|kid|head\s+stuck)\b", text, re.IGNORECASE) and re.search(r"\b(stuck|trapped|dropped|fell|fall|falling|unconscious|choking|bleeding|drown|hurt|balcony|toilet)\b", text, re.IGNORECASE):
            where = f" at {loc}" if has_loc else ""
            loc_req = "" if has_loc else f"\n\n*Please reply with your block and unit immediately.*{_LEVELS_GUIDE}"
            return (
                f"CRITICAL LIFE-SAFETY EMERGENCY — ESCALATED TO LIVE HUMAN AGENT & ON-DUTY MANAGER{where}!\n"
                f"Automated bot handling is bypassed; a human estate officer is reviewing this report now.\n\n"
                f"1. **DIAL 995 (SCDF AMBULANCE & RESCUE) IMMEDIATELY**.\n"
                f"2. **CALL 24/7 GUARDHOUSE AT `+65 6789 0123`**: On-duty security has been alerted for immediate first-responder assistance.\n"
                f"3. Do not pull or wrench violently; keep airways clear and support the child's head and neck until SCDF arrives.{loc_req}"
            )

        # Fall from height / balcony / window / ledge
        if re.search(r"\b(dropped|fell|fall|falling|jumped|jumping|hanging)\b", text, re.IGNORECASE) and re.search(r"\b(balcony|window|ledge|railing|roof|parapet|stairs|staircase|building|block|tower|floor)\b", text, re.IGNORECASE):
            where = f" at {loc}" if has_loc else ""
            loc_req = "" if has_loc else f"\n\n*Please state your block and unit or drop location immediately.*{_LEVELS_GUIDE}"
            return (
                f"CRITICAL LIFE-SAFETY EMERGENCY — ESCALATED TO LIVE HUMAN AGENT & ON-DUTY MANAGER: Fall from height logged{where}!\n"
                f"Automated bot handling is bypassed; a human estate officer is reviewing this report now.\n\n"
                f"1. **DIAL 995 (SCDF AMBULANCE & RESCUE) IMMEDIATELY**.\n"
                f"2. **CALL 24/7 GUARDHOUSE AT `+65 6789 0123`**: Security patrol dispatched to provide immediate access and first aid.\n"
                f"3. Do not attempt to move an injured person unless in immediate secondary danger.{loc_req}"
            )

        # Lift entrapment
        if re.search(r"\b(elevator|lift)\b", text, re.IGNORECASE) and re.search(r"\b(trapped|stuck|dropped|broke|falling)\b", text, re.IGNORECASE):
            where = f" at {loc}" if has_loc else ""
            if not has_loc:
                return (
                    "EMERGENCY — ESCALATED TO LIVE HUMAN AGENT: Lift entrapment report received. Automated bot handling is bypassed. "
                    "Please remain calm inside; mechanical ventilation is active. "
                    "Please provide your lift or block location immediately. Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995`."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"EMERGENCY — ESCALATED TO LIVE HUMAN AGENT: Lift entrapment reported{where}. Automated bot handling is bypassed. "
                "Please remain calm inside; mechanical ventilation is active. "
                "Our emergency response team has been alerted. Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995`."
            )
        if not has_loc:
            return (
                "EMERGENCY — ESCALATED TO LIVE HUMAN AGENT: Urgent safety priority. Automated bot handling is bypassed and a human officer is reviewing this now. "
                "Please provide your unit or block location immediately. "
                "Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995` if immediate danger exists."
                f"{_LEVELS_GUIDE}"
            )
        return (
            f"EMERGENCY — ESCALATED TO LIVE HUMAN AGENT: Urgent safety priority logged for {loc}. Automated bot handling is bypassed; our on-duty security and human management team has been alerted. "
            "Call 24/7 Security Guardhouse at `+65 6789 0123` or SCDF at `995` if life or safety is in immediate danger."
        )

    # Attempt dynamic reply via Gemini Flash before falling back to instant templates
    llm_reply = _generate_gemini_flash_resident_reply(request, classification, has_loc, loc)
    if llm_reply:
        return llm_reply

    # 1.5 COMPOUND DEFECT + CLEANLINESS (e.g. "tennis court net is torn and sagging, and empty beer bottles and trash left on benches")
    has_equipment_or_repair_defect = bool(
        re.search(
            r"\b(torn|sagging|broken|snapped|cracked|damaged|damage|faulty|spoilt|loose|detached|not\s+working|won'?t\s+turn\s+on|malfunctioning|leak|leaking|flickering)\b",
            text,
            re.IGNORECASE,
        )
    )
    has_cleanliness_defect = bool(
        re.search(
            r"\b(trash|rubbish|litter|bottles|beer\s+bottles|cans|dirty|mess|messy|unclean|uncleaned|spill|spilled|filthy|debris|vomit|poop)\b",
            text,
            re.IGNORECASE,
        )
    )
    if has_equipment_or_repair_defect and has_cleanliness_defect:
        if not has_loc:
            return (
                "Noted on both the maintenance repair issue and the cleanliness concern. "
                "Please provide the facility or block location so our facilities maintenance and housekeeping teams can attend."
                f"{_LEVELS_GUIDE}"
            )
        if "tennis" in text.lower() and "net" in text.lower():
            return (
                f"Noted on both the damaged tennis court net and the trash left on the benches. "
                f"I have logged this request for our facilities maintenance team to repair the net and our housekeeping team to clear the bottles and litter at {loc}."
            )
        return (
            f"Noted on both the damaged facility equipment and the cleanliness issue at {loc}. "
            f"I have logged this request for our facilities maintenance team to carry out repairs and our housekeeping team to clear the trash and litter."
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
        if has_equipment_or_repair_defect:
            if not has_loc:
                return (
                    "Noted on the facility equipment issue. Please provide the facility name or block location so our clubhouse and facilities maintenance team can inspect."
                    f"{_LEVELS_GUIDE}"
                )
            return (
                f"Noted on the facility equipment issue. I have logged this request for our clubhouse and facilities maintenance team to inspect and rectify at {loc}."
            )
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
