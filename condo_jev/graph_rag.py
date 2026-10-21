"""
Hybrid Knowledge Graph RAG Engine for Condo Estate Management.

Combines:
1. Fast semantic/keyword seed retrieval (Unit numbers, issue types, by-law queries).
2. Bidirectional graph traversal (1-hop & 2-hop BFS across [[Wikilinks]] and backlinks).
3. Stub-note filtering & priority expansion (neighbor units, valves, SLAs).
4. Grounded synthesis with local Ollama, Gemini, or rule-based fallback.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT_DIR = ROOT / "Knowledge Base" / "vault"

STOP_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "it",
    "my",
    "our",
    "their",
    "this",
    "that",
    "with",
    "by",
    "from",
    "be",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "if",
    "then",
    "into",
    "so",
    "than",
    "too",
    "very",
    "just",
    "about",
    "there",
    "here",
    "any",
    "some",
}


@dataclass
class NoteData:
    title: str
    note_type: str
    category: str
    aliases: list[str]
    tags: list[str]
    body: str
    preview: str
    outgoing_links: list[str]
    relationships: list[tuple[str, str]]  # (RelationType, TargetTitle)
    file_path: Path

    @property
    def is_stub(self) -> bool:
        """Stub notes are auto-generated or contain less than 60 bytes of body."""
        return len(self.body.strip()) < 60 or "auto_generated" in self.tags


@dataclass
class EstateRAGResult:
    query: str
    answer: str
    seed_nodes: list[str]
    expanded_nodes: list[str]
    edges: list[tuple[str, str, str]]  # (source, relation, target)
    by_laws_cited: list[str]
    affected_infrastructure: list[str]
    emergency_contacts: list[dict[str, str]]
    grounded: bool
    context_text: str = ""


class EstateVaultGraph:
    """In-memory Knowledge Graph representation of the Condo Estate Obsidian Vault."""

    def __init__(self, vault_dir: Path | str | None = None) -> None:
        self.vault_dir = Path(vault_dir) if vault_dir else DEFAULT_VAULT_DIR
        self.notes: dict[str, NoteData] = {}
        self.title_lookup: dict[str, str] = {}  # lowercase -> canonical title
        self.alias_lookup: dict[str, str] = {}  # lowercase alias -> canonical title
        self.outgoing_edges: dict[str, set[tuple[str, str]]] = {}  # source -> {(relation, target)}
        self.incoming_edges: dict[str, set[tuple[str, str]]] = {}  # target -> {(relation, source)}
        self._load_vault()

    def _load_vault(self) -> None:
        if not self.vault_dir.exists():
            return

        for md_file in self.vault_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                note = self._parse_note(md_file, content)
                self.notes[note.title] = note
                self.title_lookup[note.title.casefold()] = note.title
                for alias in note.aliases:
                    self.alias_lookup[alias.casefold()] = note.title
            except Exception:
                continue

        # Build bidirectional edges
        for title, note in self.notes.items():
            if title not in self.outgoing_edges:
                self.outgoing_edges[title] = set()
            for rel, target in note.relationships:
                canonical_target = self.resolve_title(target)
                if canonical_target:
                    self.outgoing_edges[title].add((rel, canonical_target))
                    if canonical_target not in self.incoming_edges:
                        self.incoming_edges[canonical_target] = set()
                    self.incoming_edges[canonical_target].add((rel, title))

    def _parse_note(self, file_path: Path, content: str) -> NoteData:
        title = file_path.stem
        note_type = "Document"
        category = "General Inquiries"
        aliases: list[str] = []
        tags: list[str] = []

        body = content
        # Parse YAML frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter, body = parts[1], parts[2]
                for line in frontmatter.strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k, v = k.strip().lower(), v.strip()
                        if k == "title" and v:
                            title = v
                        elif k == "type" and v:
                            note_type = v
                        elif k == "category" and v:
                            category = v
                        elif k == "aliases":
                            # Extract aliases from [a, b, c]
                            inner = re.sub(r"[\[\]]", "", v)
                            aliases = [a.strip() for a in inner.split(",") if a.strip()]
                        elif k == "tags":
                            inner = re.sub(r"[\[\]]", "", v)
                            tags = [t.strip() for t in inner.split(",") if t.strip()]

        # Extract all [[Wikilinks]]
        wikilinks_raw = re.findall(r"\[\[(.*?)\]\]", body)
        outgoing_links = [link.split("|")[0].strip() for link in wikilinks_raw]

        # Extract structured typed relationships from '## Relationships' section
        relationships: list[tuple[str, str]] = []
        if "## Relationships" in body:
            rel_section = body.split("## Relationships", 1)[1].split("##", 1)[0]
            for line in rel_section.strip().splitlines():
                match = re.search(r"-\s*\*\*([A-Z_]+)\*\*\s*(?:→|->)\s*\[\[(.*?)\]\]", line)
                if match:
                    rel_type, target = match.group(1), match.group(2).split("|")[0].strip()
                    relationships.append((rel_type, target))

        # If no explicit relationships section, map generic links as 'LINKS_TO'
        if not relationships:
            for link in outgoing_links:
                relationships.append(("LINKS_TO", link))

        preview = body.strip().replace("\n", " ")[:200]
        return NoteData(
            title=title,
            note_type=note_type,
            category=category,
            aliases=aliases,
            tags=tags,
            body=body.strip(),
            preview=preview,
            outgoing_links=outgoing_links,
            relationships=relationships,
            file_path=file_path,
        )

    def resolve_title(self, query_title: str) -> str | None:
        """Resolve a title or alias to the canonical vault note title."""
        cleaned = query_title.strip()
        cf = cleaned.casefold()
        if cf in self.title_lookup:
            return self.title_lookup[cf]
        if cf in self.alias_lookup:
            return self.alias_lookup[cf]
        # Check unit variations (e.g. "08-04" -> "Unit #08-04" or "#08-04")
        unit_match = re.search(r"#?(\d{1,3}-\d{1,4})", cleaned)
        if unit_match:
            u_title = f"Unit #{unit_match.group(1)}"
            if u_title.casefold() in self.title_lookup:
                return self.title_lookup[u_title.casefold()]
        return None

    def search_seeds(
        self, query: str, unit_hint: str | None = None, top_k: int = 3
    ) -> list[NoteData]:
        """
        Hybrid seed selection using exact unit extraction, title matching, and BM25 term frequency.
        """
        seeds: list[NoteData] = []
        seen_titles: set[str] = set()

        # 1. Deterministic Unit Seed Injection
        unit_str = unit_hint
        if not unit_str:
            unit_re = re.search(r"#?(\d{1,2}-\d{2,4})\b", query)
            if unit_re:
                unit_str = unit_re.group(1)

        if unit_str:
            canonical_unit = self.resolve_title(unit_str)
            if canonical_unit and canonical_unit in self.notes:
                seeds.append(self.notes[canonical_unit])
                seen_titles.add(canonical_unit)

        # 2. Token BM25 Scoring
        query_tokens = [
            t.lower()
            for t in re.findall(r"\b[A-Za-z0-9#§\-]+\b", query)
            if len(t) > 1 and t.lower() not in STOP_WORDS
        ]
        if not query_tokens:
            query_tokens = [t.lower() for t in query.split() if t.lower() not in STOP_WORDS]

        scored: list[tuple[float, NoteData]] = []
        clean_query = query.strip().casefold()

        # Corpus IDF calculation
        n_docs = max(1, len(self.notes))
        idf: dict[str, float] = {}
        for token in query_tokens:
            doc_count = sum(
                1
                for n in self.notes.values()
                if token in f"{n.title} {' '.join(n.aliases)} {n.body}".lower()
            )
            idf[token] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

        for title, note in self.notes.items():
            if title in seen_titles or note.is_stub:
                continue

            score = 0.0
            title_cf = title.casefold()
            body_cf = note.body.casefold()
            aliases_cf = [a.casefold() for a in note.aliases]

            # Exact title or alias match boost
            if clean_query == title_cf or clean_query in aliases_cf:
                score += 150.0
            elif title_cf in clean_query or any(a in clean_query for a in aliases_cf if len(a) > 3):
                score += 80.0

            # Token level matches
            for token in query_tokens:
                term_idf = idf.get(token, 1.0)
                if token in title_cf or any(token in a for a in aliases_cf):
                    score += 35.0 * term_idf
                tf = len(re.findall(r"\b" + re.escape(token) + r"\b", body_cf))
                if tf > 0:
                    score += (tf / (tf + 1.5)) * 15.0 * term_idf

            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, note in scored:
            if len(seeds) >= top_k:
                break
            if note.title not in seen_titles:
                seeds.append(note)
                seen_titles.add(note.title)

        return seeds

    def expand_subgraph(
        self,
        seed_notes: list[NoteData],
        hops: int = 1,
        max_total_notes: int = 12,
    ) -> dict[str, NoteData]:
        """
        Execute Breadth-First Search (BFS) graph expansion across outgoing and incoming links.
        Guarantees that connected neighbors (contractors, risers, valves, upstairs units,
        by-laws) are actively pulled into the context window with priority.
        """

        def _entity_priority(title: str) -> int:
            if title not in self.notes:
                return 9
            n_type = self.notes[title].note_type
            priorities = {
                "Contractor": 1,
                "InternalTeam": 1,
                "Valve": 2,
                "Unit": 2,
                "ByLaw": 2,
                "Infrastructure": 3,
                "Legislation": 3,
                "DomainCategory": 4,
                "Stack": 5,
                "Building": 6,
            }
            return priorities.get(n_type, 4)

        subgraph: dict[str, NoteData] = {}
        queue: list[tuple[str, int]] = []

        for seed in seed_notes:
            subgraph[seed.title] = seed
            queue.append((seed.title, 0))

        while queue and len(subgraph) < max_total_notes:
            curr_title, current_hop = queue.pop(0)
            if current_hop >= hops:
                continue

            # Collect candidates from both outgoing edges and incoming backlinks
            candidates: set[str] = set()
            for _, target_title in self.outgoing_edges.get(curr_title, set()):
                if target_title not in subgraph and target_title in self.notes:
                    if not self.notes[target_title].is_stub:
                        candidates.add(target_title)

            for _, source_title in self.incoming_edges.get(curr_title, set()):
                if source_title not in subgraph and source_title in self.notes:
                    if not self.notes[source_title].is_stub:
                        candidates.add(source_title)

            # Sort candidate neighbors by domain actionability
            sorted_candidates = sorted(candidates, key=_entity_priority)

            for cand_title in sorted_candidates:
                if len(subgraph) >= max_total_notes:
                    break
                if cand_title not in subgraph:
                    subgraph[cand_title] = self.notes[cand_title]
                    queue.append((cand_title, current_hop + 1))

        return subgraph

    def get_subgraph_edges(self, subgraph_notes: dict[str, NoteData]) -> list[tuple[str, str, str]]:
        """Extract all directed edges (Source, Relation, Target) within the active subgraph."""
        edges: list[tuple[str, str, str]] = []
        titles = set(subgraph_notes.keys())

        for source in titles:
            for rel, target in self.outgoing_edges.get(source, set()):
                if target in titles:
                    edges.append((source, rel, target))

        return edges

    def format_context(self, subgraph_notes: dict[str, NoteData]) -> str:
        """Format the connected subgraph into clean, structured Markdown context blocks."""
        blocks: list[str] = []
        for title, note in subgraph_notes.items():
            block = (
                f"--- NOTE: [[{title}]] (Type: {note.note_type} | Category: {note.category}) ---\n"
            )
            block += f"{note.body}\n"
            blocks.append(block)
        return "\n\n".join(blocks)

    def query(
        self,
        user_query: str,
        unit_hint: str | None = None,
        hops: int = 1,
        max_notes: int = 12,
        synthesize_answer: bool = True,
    ) -> EstateRAGResult:
        """
        End-to-End Hybrid Graph RAG Query execution:
        1. Hybrid seed search (Unit extraction + BM25 keyword matching).
        2. 1-hop / 2-hop BFS graph expansion across estate topology and by-laws.
        3. Context synthesis with grounded citations, emergency actions, and penalty checks.
        """
        seeds = self.search_seeds(user_query, unit_hint=unit_hint, top_k=3)
        subgraph = self.expand_subgraph(seeds, hops=hops, max_total_notes=max_notes)
        edges = self.get_subgraph_edges(subgraph)
        context_text = self.format_context(subgraph)

        # Extract structured domain details from the retrieved subgraph
        by_laws: list[str] = [
            t for t, n in subgraph.items() if n.note_type == "ByLaw" or "By-law" in t
        ]
        infrastructure: list[str] = [
            t
            for t, n in subgraph.items()
            if n.note_type in {"Infrastructure", "Valve", "Stack", "Building", "Unit"}
        ]
        contractors: list[dict[str, str]] = []
        for t, n in subgraph.items():
            if n.note_type in {"Contractor", "InternalTeam"}:
                phone_match = re.search(r"(\+65\s*\d{4}[-\s]?\d{4}|EXT\s*\d+)", n.body)
                phone = phone_match.group(1) if phone_match else "N/A"
                sla_match = re.search(r"SLA:\s*~?(\d+)\s*mins?", n.body, re.IGNORECASE)
                sla = sla_match.group(1) if sla_match else "30"
                coverage_match = re.search(r"Coverage:\s*([^\n|]+)", n.body, re.IGNORECASE)
                coverage = coverage_match.group(1).strip() if coverage_match else "24/7"
                contractors.append(
                    {
                        "name": t,
                        "type": n.note_type,
                        "phone": phone,
                        "sla": sla,
                        "coverage": coverage,
                        "category": n.category,
                    }
                )

        # Synthesize Grounded Answer (skip expensive LLM call when only metadata is needed)
        answer = (
            self._synthesize_grounded_answer(
                query=user_query,
                unit_hint=unit_hint,
                subgraph=subgraph,
                by_laws=by_laws,
                infrastructure=infrastructure,
                contractors=contractors,
            )
            if synthesize_answer
            else ""
        )

        return EstateRAGResult(
            query=user_query,
            answer=answer,
            seed_nodes=[s.title for s in seeds],
            expanded_nodes=list(subgraph.keys()),
            edges=edges,
            by_laws_cited=by_laws,
            affected_infrastructure=infrastructure,
            emergency_contacts=contractors,
            grounded=bool(subgraph),
            context_text=context_text,
        )

    def _llm_synthesize(self, query: str, subgraph: dict[str, NoteData]) -> str | None:
        """Call Gemini Flash to synthesize an answer dynamically from retrieved note bodies."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            context = "\n\n".join(
                f"### Note: {n.title} ({n.category})\n{n.body.strip()}"
                for n in list(subgraph.values())[:5]
            )
            prompt = (
                "You are the Condo Agent estate assistant. Answer the resident's question\n"
                "accurately using ONLY the retrieved estate notes below. Mention by-laws and\n"
                "entities in [[Wikilinks]] format. Include exact numbers, hours, fees, or\n"
                "penalties if present in the notes.\n\n"
                f"Retrieved Estate Knowledge:\n{context}\n\n"
                f"Resident Question: {query}\n\nAnswer:"
            )
            for m in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]:
                try:
                    res = client.models.generate_content(model=m, contents=prompt)
                    if res.text and len(res.text.strip()) > 30:
                        return res.text.strip()
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _synthesize_grounded_answer(
        self,
        query: str,
        unit_hint: str | None,
        subgraph: dict[str, NoteData],
        by_laws: list[str],
        infrastructure: list[str],
        contractors: list[dict[str, str]],
    ) -> str:
        """
        Dynamically synthesize a grounded answer directly from retrieved subgraph notes.
        Zero hardcoded scenario templates: every section, table row, and relationship
        is extracted dynamically from the note markdown files in the vault.
        """
        if not subgraph:
            return "No matching estate policies or infrastructure found in the knowledge graph."

        # 1. Attempt dynamic LLM synthesis if Gemini API key is available
        llm_answer = self._llm_synthesize(query, subgraph)
        if llm_answer:
            return llm_answer

        # 2. Dynamic Grounded Note Extractor (Deterministic & Offline)
        q_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]

        def note_sort_key(item: tuple[str, NoteData]) -> tuple[int, int]:
            title, n = item
            if n.note_type == "ByLaw":
                priority = 0
            elif n.note_type in ["Unit", "Valve", "Riser"]:
                priority = 1
            else:
                priority = 2
            overlap = sum(1 for tok in q_tokens if tok in n.body.lower())
            return (priority, -overlap)

        sorted_items = sorted(subgraph.items(), key=note_sort_key)

        sections = []
        for title, note in sorted_items[:3]:
            body = note.body.strip()
            summary_match = re.search(r"## Summary\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL)
            summary_text = summary_match.group(1).strip() if summary_match else note.preview

            lines = body.splitlines()
            table_header = []
            table_rows = []
            in_table = False

            for line in lines:
                if "|" in line:
                    if not in_table:
                        in_table = True
                        table_header.append(line)
                    elif len(table_header) < 2:
                        table_header.append(line)
                    else:
                        table_rows.append(line)
                elif in_table:
                    in_table = False

            card = f"📋 **[[{title}]]** ({note.category})\n{summary_text}\n"
            if table_header and table_rows:
                # Dynamically rank table rows by token overlap with query
                ranked_rows = sorted(
                    table_rows,
                    key=lambda r: sum(1 for tok in q_tokens if tok in r.lower()),
                    reverse=True,
                )
                card += "\n" + "\n".join(table_header + ranked_rows[:5]) + "\n"
            sections.append(card)

        infra_relationships = []
        for title, note in sorted_items:
            for rel, target in note.relationships:
                if target in subgraph or any(
                    k in rel
                    for k in [
                        "BELOW",
                        "ABOVE",
                        "ISOLATED",
                        "FED",
                        "GOVERNS",
                        "CONNECTED",
                        "SERVICES",
                    ]
                ):
                    infra_relationships.append(f"- **[[{title}]]** `{rel}` → **[[{target}]]**")

        answer_parts = ["\n\n".join(sections)]
        if infra_relationships:
            unique_infra = list(dict.fromkeys(infra_relationships))[:4]
            answer_parts.append(
                "### 🔧 Connected Infrastructure Topology:\n" + "\n".join(unique_infra)
            )

        if contractors:
            contractor_entries = [
                f"- **[[{c['name']}]]**: `{c['phone']}` "
                f"(Coverage: {c.get('coverage', '24/7')}, SLA: {c.get('sla', '30')} mins)"
                for c in contractors[:2]
            ]
            answer_parts.append(
                "### 📞 Emergency Dispatch Contacts:\n" + "\n".join(contractor_entries)
            )

        return "\n\n".join(answer_parts)


# Singleton vault graph instance for global application use
_estate_graph_instance: EstateVaultGraph | None = None


def get_estate_vault() -> EstateVaultGraph:
    """Lazily load and cache the Estate Vault Knowledge Graph."""
    global _estate_graph_instance
    if _estate_graph_instance is None:
        _estate_graph_instance = EstateVaultGraph()
    return _estate_graph_instance
