# Model Card: Condo Jev Triage (System-One Resident Request Classifier)

## 1. Intended Use & Scope

**Condo Jev Triage** is a structured decision-making pipeline for triaging condominium residents' service requests using **TypeSafe AI's Jev ("System One")** paradigm. Rather than generating free-form conversational prose, the pipeline evaluates an untrusted resident request (`state`) against a parallel set of **typed questions** built from Jev's three primitives:

- **`Choice` (`department_category`)**: Selects one primary estate management category from a fixed 10-way enumeration and outputs the full probability distribution plus a concentration confidence score in $[0, 1]$.
- **`Score` (`urgency_level`)**: Evaluates operational severity along an ordered 4-level rubric (`low` $\rightarrow$ `normal` $\rightarrow$ `high` $\rightarrow$ `emergency`), returning a continuous score in $[0, 1]$ and level probabilities.
- **`Noul` (4 parallel binary Yes/No questions)**: Returns $P(\text{yes}) \in [0, 1]$ for:
  1. `life_safety_hazard` — Immediate fire, smoke, gas, trapped lift passenger, flooding, or electrical danger.
  2. `multi_intent_request` — Two or more independent requests bundled in a single message.
  3. `adversarial_or_sensitive` — Prompt-injection instructions, policy override attempts, or sensitive legal/police/PDPA allegations.
  4. `requires_human_triage` — Composite probability that a human property officer should inspect the ticket before dispatch.

**Out-of-Scope Uses:** This system is an assistive triage layer. It does not autonomously dispatch emergency services, execute financial refunds, or replace statutory MCST governance.

---

## 2. Backends & Calibration Status

| Backend | Identifier | How It Evaluates Jev Primitives | Calibration Status |
| --- | --- | --- | --- |
| **TypeSafe Jev Cloud API** | `jev_cloud` | Calls `TypeSafeClient().system_one(state=..., questions=...)` via official `typesafe-sdk` (`>=0.7.4`). | `calibrated_jev_api` |
| **Offline Jev Simulator** | `jev_local` | Multinomial cross-entropy TF-IDF (word 1–2 n-grams + char 3–5 n-grams) logistic regression + deterministic `Score`/`Noul` estimators. Works 100% offline with zero API keys. | `platt_calibrated_local` |
| **Local Ollama Jev Emulator** | `ollama_jev` | Prompts a local Ollama model (e.g., `gemma3:4b`) to emit the Jev `Choice` / `Score` / `Noul` JSON contract with Pydantic validation and verbatim evidence verification. | `uncalibrated_llm_self_report` (Always routed to human review) |

---

## 3. Datasets & Integrity Verification

All data in `data/` is **100% synthetic** and contains no personal data:

- **`data/train.csv`**: 160 synthetic resident requests (16 per category across all 10 categories).
  - SHA-256: `4a9bd1f114eed1f4f346b36e402c36fd0f4c059c9a2b4cc15a0942d9d60cf8d2`
- **`data/evaluation.csv`**: 60 disjoint synthetic evaluation requests spanning five evaluation slices:
  - `standard` (45 cases across the 9 specific operational categories)
  - `ambiguous` (5 vague or out-of-scope `other` cases)
  - `emergency` (4 life-safety / infrastructure emergencies)
  - `multi_intent` (3 compound tickets combining two departments)
  - `adversarial` (3 prompt-injection and legal escalation cases)
  - SHA-256: `b2e455836f49b8067c36cdc8bb328842f6b20657b4a27c230b766e03bd7bb6f4`

`verify_no_train_eval_leakage()` enforces zero normalized text overlap between `train.csv` and `evaluation.csv` prior to every evaluation run.

---

## 4. Benchmark Evaluation Results (`reports/jev-evaluation-report.json`)

Evaluated across all 60 cases in `data/evaluation.csv` using the default `JevTriageService` (`LocalJevSimulatorBackend`, `confidence_threshold=0.55`, `noul_review_threshold=0.50`):

| Metric | Value | Notes |
| --- | --- | --- |
| **Category Accuracy (`Choice`)** | **96.67%** (58 / 60) | Perfect 1.00 F1 across 8 of 10 categories |
| **Priority Accuracy (`Score`)** | **100.0%** (60 / 60) | Across `low`, `normal`, `high`, and `emergency` |
| **Emergency Hazard Recall** | **100.0%** (4 / 4) | Caught deterministically by `safety_gate` |
| **Multi-Intent `Noul` Recall** | **100.0%** (3 / 3) | `nouls["multi_intent_request"] >= 0.50` |
| **Adversarial `Noul` Recall** | **100.0%** (3 / 3) | `nouls["adversarial_or_sensitive"] >= 0.50` |
| **Review Routing Accuracy** | **83.33%** (50 / 60) | Conservative human-in-the-loop escalation |
| **Multi-Class Brier Score** | **0.0175** | Lower is better (0.00 = perfect) |
| **Mean Confidence (Correct vs. Incorrect)** | **0.7168 vs. 0.4340** | Strong separation across the `0.55` policy threshold |
| **Mean Pipeline Latency** | **2.62 ms** | Per-ticket end-to-end LangGraph execution |

---

## 5. Limitations & Safety Considerations

1. **Synthetic Domain Distribution:** Both training and evaluation splits are synthetic English condo messages. Real-world estate traffic includes multilingual code-switching, typos, and estate-specific facility names that require periodic domain recalibration.
2. **Uncalibrated Generative LLMs:** When using the `ollama_jev` backend, self-reported confidence numbers are not true probability distributions; the `jev_policy_router` therefore requires human review on all LLM-emulated outputs.
3. **Fail-Safe Escalation:** Any provider timeout, missing API key, or schema validation failure routes the ticket to `other` (`Triage Supervisor Desk`) with `needs_human_review=True` so no resident request is silently dropped.
