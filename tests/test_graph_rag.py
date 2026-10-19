"""
Unit & Integration Tests for Condo Estate Hybrid Knowledge Graph RAG.
Verifies bidirectional traversal, multi-hop retrieval, seed resolution, and grounded synthesis.
"""

from condo_jev.graph_rag import get_estate_vault


def test_vault_loading_and_counts():
    vault = get_estate_vault()
    assert len(vault.notes) >= 20, f"Expected at least 20 notes, found {len(vault.notes)}"
    assert "Unit #08-04" in vault.notes
    assert "Unit #09-04" in vault.notes
    assert "By-law §4 - Renovation and Noise" in vault.notes
    assert "By-law §6 - Inter-Floor Water Leaks" in vault.notes
    assert "Apex 24-7 Rapid Plumbing" in vault.notes


def test_title_and_alias_resolution():
    vault = get_estate_vault()
    # Test direct resolution
    assert vault.resolve_title("Unit #08-04") == "Unit #08-04"
    # Test lowercase
    assert vault.resolve_title("unit #08-04") == "Unit #08-04"
    # Test unit variations without hash
    assert vault.resolve_title("08-04") == "Unit #08-04"
    assert vault.resolve_title("#09-04") == "Unit #09-04"
    # Test alias lookup
    assert vault.resolve_title("Renovation Rules") == "By-law §4 - Renovation and Noise"
    assert vault.resolve_title("Emergency Plumber") == "Apex 24-7 Rapid Plumbing"


def test_bidirectional_graph_edges():
    vault = get_estate_vault()
    out_edges = vault.outgoing_edges.get("Unit #08-04", set())
    targets = {target for _, target in out_edges}
    assert "Unit #09-04" in targets
    assert "Valve V-T2-08B" in targets
    assert "Riser Stack B" in targets

    # Check incoming backlinks from Unit #08-04 to Unit #09-04
    in_edges_0904 = vault.incoming_edges.get("Unit #09-04", set())
    sources_0904 = {source for _, source in in_edges_0904}
    assert "Unit #08-04" in sources_0904


def test_seed_search_with_unit_extraction():
    vault = get_estate_vault()
    # Query mentions unit number directly
    seeds = vault.search_seeds("Help! Water is leaking from ceiling in #08-04 master bath", top_k=3)
    seed_titles = [s.title for s in seeds]
    assert "Unit #08-04" in seed_titles
    assert any("Water" in s or "Plumbing" in s or "By-law §6" in s for s in seed_titles)


def test_programmatic_multi_hop_expansion():
    """
    CRITICAL MULTI-HOP TEST:
    Given a query that ONLY mentions '#08-04' and 'ceiling leak',
    prove that graph expansion actively pulls:
    1. Upstairs unit (#09-04)
    2. Isolation valve (Valve V-T2-08B)
    3. Riser line (Riser Stack B)
    4. Emergency Plumber (Apex 24-7 Rapid Plumbing)
    even though the prompt DID NOT contain any of those words!
    """
    vault = get_estate_vault()
    result = vault.query("Water is leaking from ceiling in #08-04", hops=2, max_notes=12)

    assert "Unit #08-04" in result.expanded_nodes
    assert "Unit #09-04" in result.expanded_nodes, "Upstairs unit must be traversed via graph edge"
    assert "Valve V-T2-08B" in result.expanded_nodes, "Isolation valve must be traversed"
    assert "Riser Stack B" in result.expanded_nodes, "Plumbing riser must be traversed"

    # Verify that edges connect them
    edge_pairs = {(s, t) for s, _, t in result.edges}
    assert ("Unit #08-04", "Unit #09-04") in edge_pairs or (
        "Unit #08-04",
        "Riser Stack B",
    ) in edge_pairs

    # Verify grounded answer synthesis
    assert "Unit #09-04" in result.answer
    assert "Valve V-T2-08B" in result.answer
    assert "BMSMA Section 101" in result.answer or "By-law §6" in result.answer
    assert "Apex 24-7 Rapid Plumbing" in result.answer or "+65 6789-0123" in result.answer


def test_weekend_drilling_by_law_multi_hop():
    """
    Test that asking about Saturday drilling retrieves By-law §4 and $500 penalty,
    even without the resident mentioning 'By-law' or 'penalty'.
    """
    vault = get_estate_vault()
    result = vault.query("Can my contractor do drilling tomorrow Saturday?")

    assert "By-law §4 - Renovation and Noise" in result.expanded_nodes
    assert "$500" in result.answer
    assert "strictly prohibited" in result.answer.lower()
    assert "Monday to Friday" in result.answer


def test_bbq_pit_booking_guidelines():
    vault = get_estate_vault()
    result = vault.query("How much is the deposit to book BBQ Pit 1 and what are the hours?")

    assert "By-law §8 - Facility Booking and Amenities" in result.expanded_nodes
    assert "$50" in result.answer
    assert "10:00 AM" in result.answer or "Session 1" in result.answer


def test_parking_wheel_clamp_policy():
    vault = get_estate_vault()
    result = vault.query(
        "My visitor parked overnight in a resident lot, is there a wheel clamp fee?"
    )

    assert "By-law §12 - Security and Access Control" in result.expanded_nodes
    assert "$100" in result.answer
    assert "wheel" in result.answer.lower()


def test_dynamic_extraction_from_arbitrary_new_bylaw(tmp_path):
    """
    Integrity Verification Test:
    Proves that the RAG engine extracts arbitrary text and tables from brand new notes
    with zero prior knowledge or hardcoded keyword branches.
    """
    from condo_jev.graph_rag import EstateVaultGraph

    fake_note = tmp_path / "By-law §99 - Trampoline Acrobatics.md"
    fake_note.write_text(
        "---\n"
        "title: By-law §99 - Trampoline Acrobatics\n"
        "type: ByLaw\n"
        "category: Facility Booking\n"
        "---\n"
        "## Summary\n"
        "Residents must register all trampoline acrobatics.\n\n"
        "| Activity | Security Deposit | Permitted Hours |\n"
        "| :--- | :--- | :--- |\n"
        "| Trampoline Jumps | $875 deposit | 03:30 AM to 04:45 AM |\n",
        encoding="utf-8",
    )

    custom_vault = EstateVaultGraph(vault_dir=tmp_path)
    result = custom_vault.query("What is the trampoline deposit and permitted hours?")
    assert "$875" in result.answer
    assert "03:30 AM" in result.answer
