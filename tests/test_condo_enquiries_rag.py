"""
Unit tests for the Condominium Enquiries RAG Knowledge Base.
Verifies FAQ retrieval accuracy, knowledge base responses, and staff enquiry logging.
"""

import pytest
from condo_jev.condo_enquiries_rag import (
    KNOWLEDGE_BASE,
    answer_condo_enquiry,
    format_staff_enquiry_log,
)


def test_knowledge_base_integrity():
    assert len(KNOWLEDGE_BASE) >= 10
    for article in KNOWLEDGE_BASE:
        assert article["article_id"]
        assert article["category"]
        assert article["title"]
        assert len(article["keywords"]) > 0
        assert len(article["questions"]) > 0
        assert article["content"]


@pytest.mark.parametrize(
    "query,expected_article_id",
    [
        ("what are the swimming pool hours?", "FAC-POOL"),
        ("when does the pool open and close?", "FAC-POOL"),
        ("is the gym open 24 hours?", "FAC-GYM"),
        ("how do i access the gym?", "FAC-GYM"),
        ("how do i book the bbq pit?", "FAC-BBQ"),
        ("what are the bbq booking charges and slots?", "FAC-BBQ"),
        ("how to book tennis court?", "FAC-TENNIS"),
        ("how much is function room rental in clubhouse?", "FAC-CLUBHOUSE"),
        ("what are the visitor parking charges?", "PARK-VISITOR"),
        ("is visitor parking free for guests?", "PARK-VISITOR"),
        ("where do visitors park overnight?", "PARK-VISITOR"),
        ("can i register a second vehicle for season parking?", "PARK-SEASON"),
        ("where are the ev charging stations located?", "PARK-SEASON"),
        ("what are the permitted renovation hours on saturday?", "RENO-HOURS"),
        ("can contractors do noisy work on weekend?", "RENO-HOURS"),
        ("where can i throw old sofa or bulky mattress?", "WASTE-BULKY"),
        ("how to pay maintenance fees via paynow?", "FEE-PAYMENT"),
        ("what is the paynow uen for mcst dues?", "FEE-PAYMENT"),
        ("are dogs allowed in the condo and pool?", "PET-RULES"),
        ("what is the pet policy for residents?", "PET-RULES"),
        ("i lost my access card how to get replacement?", "CARD-ACCESS"),
        ("how much is a new keycard?", "CARD-ACCESS"),
        ("where are the parcel lockers located?", "PARCEL-LOCKER"),
        ("what are the estate management office opening hours?", "OFFICE-CONTACT"),
        ("what is the security guardhouse phone number?", "OFFICE-CONTACT"),
    ],
)
def test_answer_condo_enquiry_matches(query: str, expected_article_id: str):
    res = answer_condo_enquiry(query)
    assert res is not None, f"Failed to match query: '{query}'"
    assert res["answered"] is True
    assert res["article_id"] == expected_article_id
    assert "reply" in res
    assert "Emerald Bay Residences" in res["reply"]
    # Verify zero emojis in answer
    for ch in res["reply"]:
        code = ord(ch)
        assert not (0x1F300 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF), f"Found emoji in reply: {ch}"


def test_answer_condo_enquiry_rejects_maintenance_issues():
    # Real maintenance or noise issues should NOT be intercepted by the RAG FAQ
    assert answer_condo_enquiry("there's a water leak in my kitchen") is None
    assert answer_condo_enquiry("corridor light is flickering outside #08-14") is None
    assert answer_condo_enquiry("pipe burst in basement 1") is None
    assert answer_condo_enquiry("party music is very loud and noisy") is None
    assert answer_condo_enquiry("lift B is stuck with someone inside") is None


def test_format_staff_enquiry_log():
    query = "what are the visitor parking charges?"
    res = answer_condo_enquiry(query)
    assert res is not None

    log = format_staff_enquiry_log(
        resident_name="Nick Fong",
        resident_username="nnickuhh",
        resident_id="12345678",
        query=query,
        answer_result=res,
    )
    assert "*[RESIDENT ENQUIRY LOG]*" in log
    assert "Nick Fong (@nnickuhh)" in log
    assert "Visitor Parking" in log
    assert "Handled Automatically by Bot" in log
    assert query in log


def test_get_article_by_id():
    from condo_jev.condo_enquiries_rag import get_article_by_id
    art = get_article_by_id("FAC-POOL")
    assert art is not None
    assert art["article_id"] == "FAC-POOL"
    assert "Swimming Pool" in art["title"]
    assert "Operating Hours" in art["reply"]

    assert get_article_by_id("NONEXISTENT-123") is None


def test_categories_helpers():
    from condo_jev.condo_enquiries_rag import get_all_categories, get_articles_by_category
    cats = get_all_categories()
    assert len(cats) >= 4
    assert "Facilities & Amenities" in cats

    fac_arts = get_articles_by_category("Facilities & Amenities")
    assert len(fac_arts) >= 3
    assert any(a["article_id"] == "FAC-POOL" for a in fac_arts)


def test_bbq_cleanliness_complaint_matching():
    query = "the person before me that booked the bbq pit did not clean up"
    res = answer_condo_enquiry(query)
    assert res is not None
    assert res["answered"] is True
    assert res["article_id"] == "FAC-BBQ-CLEAN"
    assert "Deposit Forfeiture" in res["reply"]
    assert "$50" in res["reply"]
    assert "cleaners" in res["reply"].lower()

    # Also test variations
    for var in (
        "the user before that did not clean up the bbq pit",
        "previous person left the bbq pit dirty",
        "what happens if previous user did not clean the bbq pit",
    ):
        r = answer_condo_enquiry(var)
        assert r is not None
        assert r["article_id"] == "FAC-BBQ-CLEAN"


