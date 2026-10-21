"""
Condominium Resident Enquiries RAG (Retrieval-Augmented Knowledge Base).
Provides authoritative mock answers for general condo inquiries (facilities, visitor parking,
BBQ bookings, renovation hours, maintenance fees, pet by-laws, and management office).
"""

from __future__ import annotations

import re
from typing import Any

# Mock Condominium Identity
CONDO_NAME = "Emerald Bay Residences"
MCST_NO = "4521"
ESTATE_OFFICE_HOURS = "Monday to Friday 9:00 AM – 6:00 PM, Saturday 9:00 AM – 1:00 PM (Closed Sundays & Public Holidays)"
ESTATE_OFFICE_PHONE = "+65 6789 0125"
ESTATE_OFFICE_EMAIL = "enquiries@emeraldbay.sg"
GUARDHOUSE_PHONE = "+65 6789 0123"

# Comprehensive Mock Knowledge Base
KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "article_id": "FAC-POOL",
        "category": "Facilities & Amenities",
        "title": "Swimming Pool Hours & Rules",
        "keywords": [
            "swimming pool", "pool", "lap pool", "wading pool", "pool hours", "pool timing",
            "swim", "pool open", "pool close", "pool maintenance", "jacuzzi"
        ],
        "questions": [
            "what are the pool hours",
            "what time does the pool open",
            "what time does the swimming pool close",
            "pool maintenance schedule",
            "can i bring guests to the pool",
            "swimming pool rules",
        ],
        "summary": "Pool open 6:00 AM to 10:00 PM daily. Maintenance every Tuesday 8:00 AM – 12:00 PM.",
        "content": (
            "• Operating Hours: 6:00 AM – 10:00 PM daily.\n"
            "• Maintenance Closure: Every Tuesday from 8:00 AM to 12:00 PM (strictly closed for chlorination and cleaning).\n"
            "• Location: Central Courtyard, Ground Level.\n"
            "• Rules:\n"
            "  - Proper swimwear is required at all times.\n"
            "  - No footwear, glassware, food, or alcoholic beverages on the pool deck.\n"
            "  - Children under 12 years must be supervised by an adult at all times.\n"
            "  - Maximum of 4 guests per resident unit."
        ),
    },
    {
        "article_id": "FAC-GYM",
        "category": "Facilities & Amenities",
        "title": "Gymnasium (Gym) Access & Hours",
        "keywords": [
            "gym", "gymnasium", "fitness centre", "fitness center", "workout", "weights",
            "treadmill", "gym hours", "gym timing", "gym 24 hours", "gym open"
        ],
        "questions": [
            "what are the gym hours",
            "is the gym open 24 hours",
            "how to access the gym",
            "can i bring guests to gym",
            "gym location",
        ],
        "summary": "Gym is open 24/7 for residents with keycard access at Clubhouse Level 2.",
        "content": (
            "• Operating Hours: 24 Hours / 7 Days a week.\n"
            "• Location: Clubhouse, Level 2.\n"
            "• Access: Resident keycard or key fob required for entry.\n"
            "• Guidelines:\n"
            "  - Clean athletic shoes and a workout towel are mandatory.\n"
            "  - Please wipe down all machines and return weights after use.\n"
            "  - Strictly for residents aged 16 and above. Due to insurance requirements, non-resident guests are not permitted in the gym."
        ),
    },
    {
        "article_id": "FAC-BBQ",
        "category": "Facilities & Amenities",
        "title": "BBQ Pit Booking Procedures & Fees",
        "keywords": [
            "bbq", "barbecue", "bbq pit", "bbq pits", "book bbq", "bbq booking",
            "bbq fee", "bbq slot", "bbq slots", "bbq timing", "grill"
        ],
        "questions": [
            "how do i book the bbq pit",
            "bbq pit booking charges",
            "bbq time slots",
            "how much is bbq booking",
            "bbq rules and guest limit",
        ],
        "summary": "Pits 1–4 bookable up to 30 days in advance. Lunch ($10) and Dinner ($15) sessions.",
        "content": (
            "• BBQ Pits: Pits 1 to 4 (located beside the Lap Pool and Pavilion).\n"
            "• Booking Sessions:\n"
            "  - Lunch Session: 11:00 AM – 3:00 PM ($10 booking fee)\n"
            "  - Dinner Session: 5:00 PM – 10:00 PM ($15 booking fee)\n"
            "• Deposit: A refundable $50 cleaning deposit is required at time of booking.\n"
            "• Capacity: Maximum of 15 guests per pit booking.\n"
            "• Quiet Hours: Music, amplifiers, and loud socializing must cease strictly by 10:00 PM.\n"
            "• How to Book: Book up to 30 days in advance via the resident portal or at the Estate Office."
        ),
    },
    {
        "article_id": "FAC-BBQ-CLEAN",
        "category": "Facilities & Cleanliness",
        "title": "BBQ Pit Cleanliness & Deposit Forfeiture Policy",
        "keywords": [
            "bbq clean", "bbq not clean", "clean up", "did not clean", "did not clean up",
            "dirty bbq", "dirty pit", "uncleaned", "previous user", "person before",
            "user before", "deposit forfeiture", "bbq mess", "left trash",
            "left dirty", "bbq dirty", "clean the pit", "clean bbq"
        ],
        "questions": [
            "the person before me that booked the bbq pit did not clean up",
            "the user before that did not clean up the bbq pit",
            "the person before me did not clean up the bbq pit",
            "the previous user didn't clean the bbq pit",
            "previous person left the bbq pit dirty",
            "what happens if previous user did not clean the bbq pit",
            "bbq pit is dirty from previous booking",
            "how to report an uncleaned bbq pit",
            "someone left the bbq pit dirty",
            "who cleans the bbq pit if previous person didn't",
            "bbq pit cleaning deposit forfeiture",
        ],
        "summary": "$50 cleaning deposit of previous user is forfeited. Housekeeping cleaners dispatched upon report.",
        "content": (
            "• Deposit Forfeiture: The previous user's $50 cleaning deposit will be forfeited following CCTV and booking log review.\n"
            "• Cleaners Dispatched: Our duty cleaners will be sent immediately to scrub and sanitize the pit for your booking."
        ),
    },
    {
        "article_id": "FAC-TENNIS",
        "category": "Facilities & Amenities",
        "title": "Tennis Court Booking & Operating Hours",
        "keywords": [
            "tennis", "tennis court", "tennis courts", "racquet", "book tennis",
            "tennis timing", "tennis hours", "court booking"
        ],
        "questions": [
            "how to book tennis court",
            "tennis court hours",
            "tennis court fee",
            "tennis court rules",
        ],
        "summary": "Courts open 7:00 AM – 10:00 PM daily. Free for residents, max 2 hours per week.",
        "content": (
            "• Operating Hours: 7:00 AM – 10:00 PM daily (floodlights turn off at 10:00 PM sharp).\n"
            "• Location: Level 1, adjacent to Block C.\n"
            "• Booking: Complimentary for residents. Each unit may book up to 2 hours per week.\n"
            "• Attire: Non-marking tennis shoes must be worn on courts.\n"
            "• Reservations: Reserve up to 7 days in advance via the resident portal."
        ),
    },
    {
        "article_id": "FAC-CLUBHOUSE",
        "category": "Facilities & Amenities",
        "title": "Function Room Rental & Clubhouse",
        "keywords": [
            "clubhouse", "function room", "event hall", "party room", "rent function room",
            "clubhouse booking", "function room booking"
        ],
        "questions": [
            "how to rent the function room",
            "function room charges and capacity",
            "clubhouse hours",
        ],
        "summary": "Function room capacity 40 pax. Rental $50 per 4-hour block plus $100 deposit.",
        "content": (
            "• Operating Hours: 9:00 AM – 10:00 PM daily.\n"
            "• Capacity: Up to 40 persons.\n"
            "• Rental Charges: $50 per 4-hour block (plus $100 refundable damage/cleaning deposit).\n"
            "• Facilities Included: Refrigerator, microwave oven, tables, chairs, and AV screen.\n"
            "• Booking: Applications must be submitted to the Estate Office at least 5 working days in advance."
        ),
    },
    {
        "article_id": "PARK-VISITOR",
        "category": "Carpark & Access",
        "title": "Visitor Parking Rates & Rules",
        "keywords": [
            "visitor parking", "visitor car", "parking rate", "parking fee", "parking charges",
            "visitor lot", "guest parking", "overnight parking", "car park charges", "carpark rates"
        ],
        "questions": [
            "what are the visitor parking charges",
            "how much is visitor parking",
            "where do visitors park",
            "overnight parking rate for guests",
            "is visitor parking free",
        ],
        "summary": "Basement 2 (Lots V01–V45). First 2 hours free, $2/hr subsequent, $15 overnight.",
        "content": (
            "• Location: Basement 2 (B2), designated Visitor Lots V01 to V45.\n"
            "• Parking Tariff:\n"
            "  - First 2 hours: Free (Complimentary)\n"
            "  - Subsequent hours: $2.00 per hour (or part thereof)\n"
            "  - Overnight Surcharge (12:00 AM – 7:00 AM): Flat rate of $15.00\n"
            "• Payment: CashCard or NETS FlashPay at exit gantry.\n"
            "• Note: Visitor parking is on a first-come, first-served basis. Parking in resident season lots will result in wheel clamping ($100 release fee)."
        ),
    },
    {
        "article_id": "PARK-SEASON",
        "category": "Carpark & Access",
        "title": "Resident Season Parking & EV Charging",
        "keywords": [
            "season parking", "resident parking", "carpark lot", "iu", "in-vehicle unit",
            "second car", "ev charging", "electric vehicle", "charger"
        ],
        "questions": [
            "how to register for season parking",
            "second car parking fee",
            "do you have ev charging stations",
            "where are the ev chargers",
        ],
        "summary": "1st car complimentary with IU registration. 2nd car $80/month. 4 EV chargers at B1.",
        "content": (
            "• 1st Vehicle: Complimentary season parking per unit with IU registered at Estate Office.\n"
            "• 2nd Vehicle: Subject to lot availability, $80.00 per month.\n"
            "• EV Charging Stations: 4 SP Mobility EV fast chargers located at Basement 1 (B1 Lots 12 to 15).\n"
            "  - Charging rates are billed directly through the SP App.\n"
            "  - Vehicles must be relocated immediately after charging completes."
        ),
    },
    {
        "article_id": "RENO-HOURS",
        "category": "Renovation & Moving",
        "title": "Permitted Renovation & Moving Hours",
        "keywords": [
            "renovation", "renovation hours", "renovation timing", "noisy work", "drilling",
            "hacking", "weekend renovation", "contractor", "moving", "moving hours", "lift padding"
        ],
        "questions": [
            "what are the allowed renovation hours",
            "can contractors work on saturday",
            "can i do renovation on sunday",
            "when can noisy works take place",
            "how to apply for renovation permit",
            "moving in and out rules",
        ],
        "summary": "Mon–Fri 9:00 AM – 5:00 PM. Sat 9:00 AM – 1:00 PM (no noisy works). Strictly none on Sundays/PH.",
        "content": (
            "• Permitted Renovation Hours:\n"
            "  - Monday to Friday: 9:00 AM – 5:00 PM\n"
            "  - Saturdays: 9:00 AM – 1:00 PM (strictly LIGHT work only; NO drilling, hacking, or tiling)\n"
            "  - Sundays & Public Holidays: Strictly NO renovation works allowed\n"
            "• Renovation Permit: Contractors must submit a Renovation Application and place a $1,000 security deposit with the Estate Office prior to commencing work.\n"
            "• Moving In/Out: Allowed Monday to Saturday 9:00 AM – 5:00 PM. Lift padding must be booked 3 days in advance ($200 refundable deposit)."
        ),
    },
    {
        "article_id": "WASTE-BULKY",
        "category": "Waste & Cleanliness",
        "title": "Bulky Waste Disposal & Bin Centre",
        "keywords": [
            "bulky waste", "bulky item", "throw sofa", "furniture disposal", "mattress",
            "bin centre", "bin center", "rubbish chute", "trash disposal", "large items"
        ],
        "questions": [
            "where can i throw old furniture",
            "how to dispose bulky items",
            "where is the bin centre",
            "free bulky disposal dates",
        ],
        "summary": "Bin Centre at B1 next to Block B ramp. Free bulky disposal on 1st & 3rd Saturday of the month.",
        "content": (
            "• Household Waste: Use the refuse chute on your floor. Waste must be securely bagged.\n"
            "• Bulky Waste Disposal:\n"
            "  - Location: Estate Bin Centre at Basement 1 (B1), beside the Block B vehicle ramp.\n"
            "  - Free Disposal Schedule: 1st and 3rd Saturday of every month (9:00 AM – 12:00 PM).\n"
            "  - Items accepted: Old mattresses, furniture, beds, desks, and small appliances.\n"
            "• Important: Dumping bulky items at common lift lobbies, corridors, or chute rooms is strictly prohibited under MCST by-laws and subject to a $200 clearance penalty."
        ),
    },
    {
        "article_id": "FEE-PAYMENT",
        "category": "Billing & Maintenance Fees",
        "title": "Maintenance Fees & Payment Methods",
        "keywords": [
            "maintenance fee", "maintenance fees", "mcst dues", "paynow", "giro",
            "invoice", "billing", "bill", "payment method", "pay fees", "sinking fund"
        ],
        "questions": [
            "how do i pay maintenance fees",
            "when are maintenance fees due",
            "what is the paynow uen for condo fees",
            "how to set up giro for maintenance fees",
            "who to contact for billing enquiries",
        ],
        "summary": "Due quarterly on 1st of Jan/Apr/Jul/Oct. PayNow to UEN 202612345M (ref: Unit Number) or GIRO.",
        "content": (
            "• Billing Cycle: Invoiced quarterly (due on the 1st of January, April, July, and October).\n"
            "• Payment Methods:\n"
            "  1. PayNow (Instant):\n"
            "     - PayNow UEN: `202612345M` (Entity Name: MCST Plan No. 4521)\n"
            "     - Reference Field: Enter your Block and Unit Number (e.g. `B-0814`)\n"
            "  2. Interbank GIRO: Deduction on the 15th of the billing month. Application forms available at the Estate Office.\n"
            "  3. Crossed Cheque: Payable to 'MCST Plan No. 4521', deposited in the Management Office drop box.\n"
            "• Billing Enquiries: Contact Finance Desk at `+65 6789 0125` (Option 2) or email `accounts@emeraldbay.sg`."
        ),
    },
    {
        "article_id": "PET-RULES",
        "category": "Estate By-Laws",
        "title": "Pet By-Laws & Guidelines",
        "keywords": [
            "pet", "pets", "dog", "dogs", "cat", "cats", "pet rules", "pet policy",
            "animals", "walking dog", "barking"
        ],
        "questions": [
            "what are the pet rules in the condo",
            "are dogs allowed in the condo",
            "can i bring my dog to the pool",
            "can i bring my dog to the clubhouse",
            "can i bring my dog to the gym",
            "are pets allowed in common areas",
            "how many pets can i keep",
        ],
        "summary": "Max 2 small pets per unit. Leash required in common areas. Strictly no pets at pool/gym/BBQ.",
        "content": (
            "• Pet Limit: Maximum of 2 small domestic pets (dogs or cats) permitted per unit.\n"
            "• Leash Rule: Dogs must be kept on a leash at all times while in estate common grounds.\n"
            "• Restricted Zones: Pets are strictly prohibited in the Swimming Pool, Pool Deck, Gymnasium, Tennis Courts, and BBQ Pavilions.\n"
            "• Sanitation: Pet owners must immediately clean up all animal waste. Failure to do so incurs an administrative cleaning fee of $100."
        ),
    },
    {
        "article_id": "CARD-ACCESS",
        "category": "Access & Security",
        "title": "Access Cards, Key Fobs & Intercom",
        "keywords": [
            "access card", "keycard", "key fob", "lost card", "replacement card",
            "intercom", "door access", "side gate access", "lift access"
        ],
        "questions": [
            "how to get replacement access card",
            "how much is a new access card",
            "my access card is not working",
            "how to add my number to the intercom",
        ],
        "summary": "Replacement cards $30 each at Estate Office. Bring proof of tenancy/ownership.",
        "content": (
            "• Card Allocation: Each unit receives 4 complimentary access cards upon key handover.\n"
            "• Replacement & Additional Cards:\n"
            "  - Fee: $30.00 per card (non-refundable).\n"
            "  - Maximum Limit: Up to 6 active cards per unit.\n"
            "  - Process: Visit Estate Management Office with proof of residency (SP bill or tenancy agreement).\n"
            "• Intercom Registration: Provide your mobile number to the Estate Office to receive visitor calls directly to your phone."
        ),
    },
    {
        "article_id": "PARCEL-LOCKER",
        "category": "Facilities & Amenities",
        "title": "Parcel Lockers & Courier Delivery",
        "keywords": [
            "parcel", "parcels", "delivery", "courier", "smart locker", "parcel locker",
            "package", "shopee", "lazada", "amazon"
        ],
        "questions": [
            "where are the parcel lockers located",
            "how does parcel locker delivery work",
            "where can couriers drop packages",
        ],
        "summary": "ParcelPoint smart lockers at Clubhouse Level 1. SMS PIN provided by courier.",
        "content": (
            "• Location: Clubhouse Level 1, adjacent to the Main Guardhouse entrance.\n"
            "• How it Works: Delivery couriers deposit parcels into ParcelPoint smart lockers. An SMS notification with a secure one-time collection PIN is sent to your registered phone number.\n"
            "• Collection Window: Please collect packages within 48 hours to ensure locker availability for fellow residents.\n"
            "• Bulky Parcels: Large items exceeding locker dimensions must be received in person."
        ),
    },
    {
        "article_id": "OFFICE-CONTACT",
        "category": "Estate Management",
        "title": "Estate Management Office & Guardhouse Contacts",
        "keywords": [
            "management office", "estate office", "guardhouse", "security desk", "contact number",
            "phone number", "office hours", "opening hours", "mcst office", "property manager"
        ],
        "questions": [
            "what are the management office opening hours",
            "how to contact the guardhouse",
            "management office phone number",
            "where is the estate office located",
        ],
        "summary": "Clubhouse Level 1. Mon–Fri 9am–6pm, Sat 9am–1pm. Tel: +65 6789 0125. Guardhouse 24/7: +65 6789 0123.",
        "content": (
            "• Location: Clubhouse Level 1 (beside Function Room).\n"
            f"• Operating Hours: {ESTATE_OFFICE_HOURS}.\n"
            f"• Office Phone: `{ESTATE_OFFICE_PHONE}`\n"
            f"• Office Email: `{ESTATE_OFFICE_EMAIL}`\n"
            f"• 24/7 Security Guardhouse: `{GUARDHOUSE_PHONE}`\n"
            "• Lift Emergency Hotline: `+65 6789 0124`\n"
            "• Emergency Services: Police `999` | SCDF (Fire & Ambulance) `995`."
        ),
    },
]


def _tokenize(text: str) -> set[str]:
    """Clean and tokenize input string into distinct normalized words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "what", "how", "when", "where",
        "who", "which", "can", "could", "do", "does", "did", "i", "we", "you", "my",
        "our", "your", "to", "for", "in", "at", "on", "of", "and", "or", "it", "this",
        "that", "there", "any", "some", "me", "us", "be", "have", "has", "get", "tell"
    }
    tokens = {t for t in cleaned.split() if len(t) > 1 and t not in stop_words}
    return tokens


def answer_condo_enquiry(query: str) -> dict[str, Any] | None:
    """
    Match resident query against mock condo knowledge base using lexical and question similarity.
    Returns structured answer dict if confident, or None if the query is not a condo enquiry.
    """
    clean_query = query.strip()
    if not clean_query or len(clean_query) < 4:
        return None

    query_tokens = _tokenize(clean_query)
    if not query_tokens:
        return None

    lowered_query = clean_query.lower()

    # Rule out obvious maintenance, repair, damage, and cleanliness issues
    maintenance_cues = {
        # Defects / Damage / Hardware faults
        "leaking", "leak", "flickering", "spoilt", "broken", "pipe burst", "stuck in lift",
        "torn", "sagging", "crack", "cracked", "damaged", "damage", "faulty", "defect",
        "malfunctioning", "not working", "won't turn on", "not turning on", "loose",
        "detached", "burnt out", "choked", "clogged", "overflowing", "spill", "spilled",
        "smell", "foul odor", "stink", "tripped", "power trip", "slipping", "jammed",
        # Noise / Disturbances
        "noisy", "music is loud", "party", "shouting", "drilling noise", "screaming", "stomping",
        # Cleanliness / Hygiene / Waste
        "trash", "rubbish", "litter", "bottles", "beer bottles", "cans", "dirty", "mess",
        "messy", "unclean", "uncleaned", "stained", "stain", "murky", "green water", "filthy",
        "dog poop", "vomit", "cigarettes", "cigarette butts", "leftover", "left on", "debris",
        # Action / Escalation requests
        "please fix", "please clean", "please clear", "need repair", "needs repair",
        "needs cleaning", "repair needed", "cleaners required", "someone left", "abandoned",
        "need it to be fixed", "need it fixed"
    }
    # Exception: "can i do noisy renovation work" or "what are renovation hours"
    is_renovation_query = any(k in lowered_query for k in ("renovation", "contractor", "contractors", "hacking", "moving", "permit", "noisy work"))
    # Exception: BBQ cleanliness deposit policy ("FAC-BBQ-CLEAN")
    is_bbq_clean_query = "bbq" in lowered_query and any(k in lowered_query for k in ("clean", "dirty", "uncleaned", "deposit", "previous"))
    if not is_renovation_query and not is_bbq_clean_query and any(cue in lowered_query for cue in maintenance_cues):
        return None

    best_match: dict[str, Any] | None = None
    best_score: float = 0.0

    # Determine if query is asking a question/inquiry vs stating a condition
    is_question = (
        "?" in query
        or any(qw in lowered_query for qw in (
            "what", "when", "how", "who", "which", "where", "can i", "is it", "are there",
            "hours", "timing", "time", "fee", "cost", "deposit", "rule", "bylaw",
            "book", "booking", "reserve", "reservation", "guideline", "policy", "permit",
            "opening", "closing", "schedule"
        ))
    )

    for article in KNOWLEDGE_BASE:
        score = 0.0

        # 1. Exact or high-overlap match with sample questions
        for q in article["questions"]:
            if q in lowered_query or lowered_query in q:
                score = max(score, 0.95)
            else:
                q_tokens = _tokenize(q)
                common_q = query_tokens.intersection(q_tokens)
                if len(common_q) >= 3:
                    overlap_ratio = len(common_q) / max(len(q_tokens), 1)
                    score = max(score, 0.60 + overlap_ratio * 0.35)

        # 2. Keyword matching
        matched_kw_count = 0
        for kw in article["keywords"]:
            if " " in kw:
                if kw in lowered_query:
                    matched_kw_count += 2
            else:
                if kw in query_tokens:
                    matched_kw_count += 1

        if matched_kw_count > 0:
            kw_score = min(0.85, 0.30 + matched_kw_count * 0.15)
            score = max(score, kw_score)

        # 3. Title overlap bonus (only grant high bonus if query is phrased as a question/inquiry)
        title_tokens = _tokenize(article["title"])
        common_title = query_tokens.intersection(title_tokens)
        if is_question:
            if len(common_title) >= 2:
                score = max(score, 0.70)
            elif len(common_title) == 1 and score > 0.30:
                score += 0.15
        else:
            if len(common_title) >= 2:
                score = max(score, 0.35)

        # Demote generic booking procedures if query is explicitly about cleanliness / dirtiness
        if article["article_id"] in ("FAC-BBQ", "FAC-TENNIS", "FAC-GYM", "FAC-POOL") and any(w in lowered_query for w in ("clean", "dirty", "uncleaned", "mess", "trash", "rubbish", "torn", "broken", "bottles")):
            score = max(0.0, score - 0.40)

        if score > best_score:
            best_score = score
            best_match = article

    # Threshold for confident answer
    if best_match and best_score >= 0.40:
        formatted_reply = (
            f"*{CONDO_NAME} — {best_match['title']}*\n"
            f"──────────────────────────────\n"
            f"{best_match['content']}\n\n"
            f"_Estate Office: {ESTATE_OFFICE_PHONE} | Guardhouse: {GUARDHOUSE_PHONE}_"
        )
        return {
            "answered": True,
            "article_id": best_match["article_id"],
            "title": best_match["title"],
            "category": best_match["category"],
            "summary": best_match["summary"],
            "reply": formatted_reply,
            "confidence": round(best_score, 2),
        }

    return None


def format_staff_enquiry_log(
    resident_name: str,
    resident_username: str | None,
    resident_id: str,
    query: str,
    answer_result: dict[str, Any],
) -> str:
    """Format a clean informational log message for the staff condo enquiries topic."""
    handle = f"@{resident_username}" if resident_username else f"User {resident_id}"
    resident_info = f"{resident_name} ({handle})" if resident_name else handle
    
    summary = answer_result.get("summary", answer_result.get("title", ""))

    lines = [
        "*[RESIDENT ENQUIRY LOG]*",
        "──────────────────────────────",
        f"• *Resident:* {resident_info}",
        f"• *Topic:* {answer_result['category']} — {answer_result['title']}",
        f"• *Status:* Handled Automatically by Bot (No Staff Action Needed)",
        f"• *Policy Summary:* {summary}",
        f"\n*Resident Asked:*\n\"{query.strip()}\"",
    ]
    return "\n".join(lines)


def get_article_by_id(article_id: str) -> dict[str, Any] | None:
    """Retrieve formatted article response by its unique ID."""
    clean_id = article_id.strip().upper()
    for art in KNOWLEDGE_BASE:
        if art["article_id"].upper() == clean_id:
            formatted_reply = (
                f"*{CONDO_NAME} — {art['title']}*\n"
                f"──────────────────────────────\n"
                f"{art['content']}\n\n"
                f"_Estate Office: {ESTATE_OFFICE_PHONE} | Guardhouse: {GUARDHOUSE_PHONE}_"
            )
            return {
                "answered": True,
                "article_id": art["article_id"],
                "title": art["title"],
                "category": art["category"],
                "summary": art["summary"],
                "reply": formatted_reply,
                "confidence": 1.0,
            }
    return None


def get_all_categories() -> list[str]:
    """Return all unique category names present in the knowledge base."""
    seen: list[str] = []
    for art in KNOWLEDGE_BASE:
        cat = art["category"]
        if cat not in seen:
            seen.append(cat)
    return seen


def get_articles_by_category(category: str) -> list[dict[str, Any]]:
    """Return all articles belonging to a specific category."""
    clean_cat = category.strip().lower()
    return [art for art in KNOWLEDGE_BASE if art["category"].lower() == clean_cat]

