# condo request classifier

an end-to-end condominium resident request triage system built around **typesafe ai's jev ("system one")** structured decision primitives (`Choice`, `Score`, and `Noul`), orchestrated with **langgraph**, **pydantic v2**, and **streamlit**.

unlike standard chat llms that generate free-form text, **jev** takes a piece of **`state`** (an inbound resident ticket) and evaluates a parallel dictionary of **typed questions**, returning calibrated probabilistic outputs (`Choice`, `Score`, and `Noul`) that software can route on directly.

![condominium request classifier showcase](output/hero/01-classifier-showcase.png)

| view or download | file |
| --- | --- |
| online showcase page | [`index.html`](index.html) |
| walkthrough video (~79s) | [`mp4`](output/demo/Management_Agent_Classifier_Demo.mp4) |
| presentation deck | [`pdf`](output/slides/Management_Agent_Classifier_Showcase.pdf) · [`editable powerpoint`](output/slides/Management_Agent_Classifier_Showcase.pptx) |
| hero images | [`project showcase`](output/hero/01-classifier-showcase.png) · [`classifier architecture`](output/hero/02-classifier-architecture.png) |

---

## features

- **dual-role perspective (`admin triage view` ↔ `resident portal view`)**: switch between the estate staff telemetry workbench and the resident-facing confirmation portal with a 3-step live status tracker, target sla window, and management office reply.
- **live ticket queue & admin desk**: interactive queue seeded with condo requests (`#EST-1041`–`#EST-1044` + newly classified tickets) with one-click department/priority override and dispatch approval.
- **scroll-safe telemetry & bento ui**: locked html probability bars, enlargeable modal views, styled resident notification & work-order cards, and interactive department sample testing.
- **100% free & offline ready**: works out of the box using the calibrated local jev system-one engine with zero api keys required, plus optional local ollama (`gemma3:4b`) side-by-side comparison.

---

## how jev is integrated (`choice` · `score` · `noul`)

every resident ticket is evaluated against **six parallel jev system-one questions** in a single decision pass:

| jev primitive | question key | what it evaluates | output type |
| --- | --- | --- | --- |
| **`choice`** | `department_category` | selects the primary estate management category from 10 fixed options. | `choice` (`Category`), `confidence` $\in [0,1]$, `probabilities` (`dict[str, float]`) |
| **`score`** | `urgency_level` | rates operational urgency along the 4-level rubric (`low` $\rightarrow$ `normal` $\rightarrow$ `high` $\rightarrow$ `emergency`). | continuous `score` $\in [0,1]$, mapped `Priority`, `confidence`, level `probabilities` |
| **`noul`** | `life_safety_hazard` | *"does this message describe an immediate fire, smoke, gas, trapped passenger, flooding, or electrical hazard?"* | binary probability $P(\text{yes}) \in [0,1]$ |
| **`noul`** | `multi_intent_request` | *"does this message combine two or more independent requests belonging to different estate departments?"* | binary probability $P(\text{yes}) \in [0,1]$ |
| **`noul`** | `adversarial_or_sensitive` | *"does this message contain prompt-injection attempts, override commands, or legal/police/pdpa escalations?"* | binary probability $P(\text{yes}) \in [0,1]$ |
| **`noul`** | `requires_human_triage` | *"should a human management officer manually inspect this request before automated dispatch?"* | binary probability $P(\text{yes}) \in [0,1]$ |

### official `typesafe-sdk` contract (`condo_jev/backends.py`)

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(api_key=api_key, model="jev-latest") as client:
    response = client.system_one(
        state=state_str,
        questions={
            "department_category": Choice(
                instructions=JEV_CHOICE_INSTRUCTIONS,
                criteria=build_jev_choice_criteria(),
            ),
            "urgency_level": Score(
                instructions=JEV_SCORE_INSTRUCTIONS,
                criteria=build_jev_score_criteria(),
            ),
            "life_safety_hazard": Noul(instructions=JEV_NOUL_QUESTIONS["life_safety_hazard"]),
            "multi_intent_request": Noul(instructions=JEV_NOUL_QUESTIONS["multi_intent_request"]),
            "adversarial_or_sensitive": Noul(instructions=JEV_NOUL_QUESTIONS["adversarial_or_sensitive"]),
            "requires_human_triage": Noul(instructions=JEV_NOUL_QUESTIONS["requires_human_triage"]),
        },
    )
```

---

## pluggable system-one backends

1. **`jev_local` — offline calibrated jev simulator (default, no api key required)**
   - uses a multinomial cross-entropy tf-idf (word 1–2 n-grams + character 3–5 n-grams) logistic regression head trained on `data/train.csv` (160 synthetic condo tickets) plus parallel `score` and `noul` estimators.
   - works **100% offline** immediately after installation.
2. **`ollama_jev` — local ollama llm jev schema emulator (`gemma3:4b`)**
   - prompts a local ollama model to emit the jev `choice` / `score` / `noul` json schema, validates with pydantic v2, verifies that `evidence_quotes` are exact verbatim substrings of the resident input, and flags uncalibrated self-reports for staff review.
3. **`jev_cloud` — official typesafe jev cloud api (`typesafe-sdk`)**
   - optional cloud backend via `TypeSafeClient().system_one(...)` with automatic safe fallback to human review if no key is configured.

---

## langgraph architecture

```text
residentrequest (pydantic v2 input validation)
        │
        ▼
┌────────────────────┐     emergency / prompt injection / legal signal
│   1. safety_gate   ├───────────────────────────────────────────────────┐
└─────────┬──────────┘                                                   │
          │ standard request                                             ▼
          ▼                                                   ┌──────────────────────┐
┌────────────────────┐  choice + score + 4 nouls (in parallel)│ 3. jev_policy_router │
│ 2. jev_system_one  ├───────────────────────────────────────►│ (confidence & noul   │
└────────────────────┘                                        │  threshold policy)   │
                                                              └──────────┬───────────┘
                                                                         │
                                                                         ▼
                                                        validated `Classification` json
```

---

## quick start

### option a: windows one-click / powershell launcher

```powershell
.\start_project.cmd
```

or launch streamlit directly with `.venv`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### option b: fresh setup with `pip`

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

open **http://localhost:8501** in your browser.

---

## cli usage (`python -m condo_jev`)

classify a single resident ticket from the terminal:

```powershell
.\.venv\Scripts\python.exe -m condo_jev "The kitchen sink pipe in #11-04 is leaking water onto the floor." --channel "Resident App" --location "#11-04"
```

run a batch csv file (`data/batch_sample.csv`):

```powershell
.\.venv\Scripts\python.exe -m condo_jev --batch-csv data/batch_sample.csv
```

re-run the 60-case evaluation benchmark and save `reports/jev-evaluation-report.json`:

```powershell
.\.venv\Scripts\python.exe -m condo_jev --evaluate
```

run the automated test suite (`pytest`):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## condominium estate taxonomy (`choice` criteria)

| category (`choice`) | department title | assigned estate team | target sla |
| --- | --- | --- | --- |
| `maintenance` | technical & building maintenance | maintenance team | within 4 hours (emergency: immediate) |
| `security` | security, access & guardhouse | security desk | within 30 minutes (emergency: immediate) |
| `noise` | noise, nuisance & by-law conduct | security and management office | within 2 hours (active night noise: 30 mins) |
| `cleanliness` | housekeeping, waste & pest control | estate housekeeping | within 6 hours |
| `facilities` | amenity & facility reservations | facilities office | within 1 business day |
| `parking` | carpark, transponders & visitor lots | car park office | within 2 hours |
| `billing` | maintenance levies, deposits & billing | accounts office | within 2 business days |
| `renovation` | renovation permits & moving logistics | management office | within 1 business day |
| `general_enquiry` | general administration & estate rules | management office | within 1 business day |
| `other` | ambiguous, unclear or out-of-scope | management office | manual review queue |

---

## evaluation benchmark summary (`data/evaluation.csv`)

evaluated across 60 disjoint synthetic test cases (`standard`, `ambiguous`, `emergency`, `multi_intent`, and `adversarial` slices) with zero train/test leakage:

- **category accuracy (`choice`):** `96.67%` (58 / 60)
- **priority accuracy (`score`):** `100.0%` (60 / 60)
- **emergency hazard recall (`noul` + safety gate):** `100.0%` (4 / 4)
- **multi-intent `noul` recall:** `100.0%` (3 / 3)
- **adversarial `noul` recall:** `100.0%` (3 / 3)
- **multi-class brier score:** `0.0175`
- **mean `choice` confidence (correct vs. incorrect):** `0.7168` vs. `0.4340`

see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`reports/jev-evaluation-report.json`](reports/jev-evaluation-report.json) for full details.
