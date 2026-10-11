import csv
import math
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError
from pydantic import ValidationError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from condo_jev.schema import (
    Category,
    JevChoiceResult,
    JevNoulResult,
    JevScoreResult,
    JevSystemOneBundle,
    LLMJevEmulationPayload,
    Priority,
    ResidentRequest,
    TriagePrediction,
)
from condo_jev.signals import (
    compute_noul_probabilities,
    compute_urgency_score_distribution,
    extract_evidence_spans,
    extract_location,
)
from condo_jev.taxonomy import (
    JEV_CHOICE_INSTRUCTIONS,
    JEV_NOUL_QUESTIONS,
    JEV_SCORE_INSTRUCTIONS,
    build_jev_choice_criteria,
    build_jev_score_criteria,
    build_ollama_jev_prompt,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = ROOT_DIR / "data" / "train.csv"


class BackendUnavailableError(RuntimeError):
    """Raised when an external model provider (TypeSafe Jev API or Ollama) cannot complete inference safely."""


class TriageBackend(Protocol):
    def predict(self, request: ResidentRequest) -> TriagePrediction: ...


def distribution_concentration_confidence(probabilities: dict[str, float]) -> float:
    """
    Compute Jev-style distribution concentration confidence in [0, 1].
    Combines top-1 vs top-2 dominance margin with normalized complement of Shannon entropy.
    """
    values = sorted((max(1e-12, float(p)) for p in probabilities.values()), reverse=True)
    total = sum(values)
    normed = [v / total for v in values]
    k = len(normed)
    if k <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in normed)
    max_entropy = math.log(k)
    entropy_concentration = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
    top1 = normed[0]
    top2 = normed[1] if k > 1 else 0.0
    margin = top1 - top2
    # Scale margin relative to a 10-class distribution where a strong pick has top1 >= 0.45
    margin_score = min(1.0, margin / 0.45)
    blended = 0.50 * top1 + 0.30 * margin_score + 0.20 * entropy_concentration
    return round(max(0.0, min(1.0, blended)), 4)


def score_float_to_priority(score: float) -> Priority:
    """Map a continuous Jev `Score` in [0, 1] onto the 4-level Priority rubric."""
    if score >= 0.82:
        return Priority.EMERGENCY
    if score >= 0.58:
        return Priority.HIGH
    if score >= 0.28:
        return Priority.NORMAL
    return Priority.LOW


def build_state_payload(request: ResidentRequest) -> str:
    """Format the resident request into a structured state string for Jev `system_one`."""
    loc = request.location or "unspecified"
    return (
        f"channel: {request.channel}\n"
        f"reported_location: {loc}\n"
        f"resident_message: {request.text}"
    )


class LocalJevSimulatorBackend:
    """
    Offline System-One Jev simulator trained on bundled synthetic condo requests.
    Uses multinomial cross-entropy TF-IDF + char n-gram logistic regression
    to produce `Choice` distributions alongside parallel `Score` and `Noul` primitives.
    """

    def __init__(self, train_csv: Path = DEFAULT_TRAIN_CSV):
        if not train_csv.exists():
            raise FileNotFoundError(f"Training dataset not found at {train_csv}")
        with train_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        texts = [row["text"] for row in rows]
        labels = [Category(row["category"]).value for row in rows]
        expected_categories = {cat.value for cat in Category}
        if set(labels) != expected_categories:
            missing = expected_categories - set(labels)
            raise ValueError(f"Training split is missing categories: {sorted(missing)}")

        self.pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                        ]
                    ),
                ),
                ("calibrated_jev_head", LogisticRegression(C=10.0, max_iter=1000, random_state=42)),
            ]
        )
        self.pipeline.fit(texts, labels)

    def predict(self, request: ResidentRequest) -> TriagePrediction:
        raw_probs = self.pipeline.predict_proba([request.text])[0]
        classes = list(self.pipeline.classes_)
        cat_probs = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(classes, raw_probs.tolist(), strict=True)
        }
        chosen_key = max(cat_probs, key=cat_probs.get)
        chosen_category = Category(chosen_key)
        choice_conf = distribution_concentration_confidence(cat_probs)

        urgency_score, priority_level, score_conf, priority_probs = compute_urgency_score_distribution(
            request.text, chosen_category
        )
        noul_map = compute_noul_probabilities(request.text, choice_conf, chosen_category)
        detected_location = extract_location(request.text, request.location)
        evidence = extract_evidence_spans(request.text)

        jev_bundle = JevSystemOneBundle(
            state_summary=build_state_payload(request),
            choices={
                "department_category": JevChoiceResult(
                    question_key="department_category",
                    instructions=JEV_CHOICE_INSTRUCTIONS,
                    choice=chosen_category,
                    confidence=choice_conf,
                    probabilities=cat_probs,
                )
            },
            scores={
                "urgency_level": JevScoreResult(
                    question_key="urgency_level",
                    instructions=JEV_SCORE_INSTRUCTIONS,
                    score=urgency_score,
                    level=priority_level,
                    confidence=score_conf,
                    probabilities=priority_probs,
                )
            },
            nouls={
                key: JevNoulResult(
                    question_key=key,
                    instructions=JEV_NOUL_QUESTIONS[key],
                    noul=val,
                )
                for key, val in noul_map.items()
            },
        )

        return TriagePrediction(
            category=chosen_category,
            priority=priority_level,
            choice_confidence=choice_conf,
            urgency_score=urgency_score,
            category_probabilities=cat_probs,
            priority_probabilities=priority_probs,
            nouls=noul_map,
            jev_bundle=jev_bundle,
            extracted_location=detected_location,
            evidence_quotes=evidence,
            rationale_note=(
                f"Calibrated local Jev simulator selected '{chosen_category.value}' "
                f"(concentration confidence {choice_conf:.0%}, top class P={cat_probs[chosen_key]:.0%})."
            ),
            backend="jev_local",
            model="jev-local-platt-v1",
            calibration_status="platt_calibrated_local",
        )


class JevCloudBackend:
    """
    Official TypeSafe AI Jev (`typesafe-sdk`) System-One backend.
    Evaluates `Choice`, `Score`, and 4 `Noul` primitives in a single parallel `system_one` request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ):
        self.api_key = (api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")).strip()
        self.model = (model or os.getenv("TYPESAFE_MODEL") or "jev-latest").strip()
        self._injected_client = client

    def _build_questions(self) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul, Score

        questions: dict[str, Any] = {
            "department_category": Choice(
                instructions=JEV_CHOICE_INSTRUCTIONS,
                criteria=build_jev_choice_criteria(),
            ),
            "urgency_level": Score(
                instructions=JEV_SCORE_INSTRUCTIONS,
                criteria=build_jev_score_criteria(),
            ),
        }
        for noul_key, instructions in JEV_NOUL_QUESTIONS.items():
            questions[noul_key] = Noul(instructions=instructions)
        return questions

    def _invoke_sdk(self, state: str, questions: dict[str, Any]) -> Any:
        if self._injected_client is not None:
            return self._injected_client.system_one(state=state, questions=questions)

        if not self.api_key:
            raise BackendUnavailableError(
                "TYPESAFE_API_KEY is not set. Add your TypeSafe AI key in .env or switch to the Offline Jev Simulator."
            )
        try:
            from typesafe_sdk import TypeSafeClient

            with TypeSafeClient(api_key=self.api_key, model=self.model) as client:
                return client.system_one(state=state, questions=questions)
        except Exception as exc:
            raise BackendUnavailableError(
                "TypeSafe Jev API request failed. Verify TYPESAFE_API_KEY and network connectivity."
            ) from exc

    def predict(self, request: ResidentRequest) -> TriagePrediction:
        state_str = build_state_payload(request)
        questions = self._build_questions()
        response = self._invoke_sdk(state=state_str, questions=questions)

        try:
            choice_obj = response.choices["department_category"]
            chosen_category = Category(str(choice_obj.choice).strip())
            raw_cat_probs = getattr(choice_obj, "probabilities", None) or {}
            cat_probs = {
                cat.value: round(float(raw_cat_probs.get(cat.value, 0.0)), 4)
                for cat in Category
            }
            if sum(cat_probs.values()) <= 0:
                cat_probs[chosen_category.value] = 1.0
            choice_conf = round(
                float(getattr(choice_obj, "confidence", distribution_concentration_confidence(cat_probs))),
                4,
            )

            score_obj = response.scores["urgency_level"]
            raw_score = float(score_obj.score)
            raw_score_probs = getattr(score_obj, "probabilities", None) or {}
            raw_legend = getattr(score_obj, "legend", None) or {}

            ordered_priorities = list(Priority)
            if raw_legend and all(isinstance(k, int) for k in raw_legend):
                sorted_keys = sorted(raw_legend.keys())
                min_k, max_k = sorted_keys[0], sorted_keys[-1]
                span = max(1.0, float(max_k - min_k))
                urgency_score = round(max(0.0, min(1.0, (raw_score - min_k) / span)), 4)
            elif raw_score > 1.0:
                urgency_score = round(max(0.0, min(1.0, raw_score / max(1, len(ordered_priorities) - 1))), 4)
            else:
                urgency_score = round(max(0.0, min(1.0, raw_score)), 4)

            priority_level = score_float_to_priority(urgency_score)
            score_conf = round(float(getattr(score_obj, "confidence", 0.80)), 4)

            priority_probs: dict[str, float] = {}
            int_keys = sorted(k for k in raw_score_probs if isinstance(k, int))
            if len(int_keys) == len(ordered_priorities):
                for idx_key, prio_enum in zip(int_keys, ordered_priorities, strict=True):
                    priority_probs[prio_enum.value] = round(float(raw_score_probs[idx_key]), 4)
            else:
                for p in ordered_priorities:
                    matched_val = 0.0
                    for k, v in raw_score_probs.items():
                        k_str = str(k)
                        if k_str == p.value or k_str.startswith(f"{p.value}:"):
                            matched_val = float(v)
                            break
                    priority_probs[p.value] = round(matched_val, 4)

            if sum(priority_probs.values()) <= 0:
                priority_probs = {
                    p.value: (1.0 if p == priority_level else 0.0) for p in ordered_priorities
                }

            noul_map: dict[str, float] = {}
            for noul_key in JEV_NOUL_QUESTIONS:
                noul_entry = response.nouls[noul_key]
                noul_val = float(getattr(noul_entry, "noul", 0.0))
                noul_map[noul_key] = round(max(0.0, min(1.0, noul_val)), 4)
        except Exception as exc:
            raise BackendUnavailableError(
                "TypeSafe Jev API returned an unexpected response structure."
            ) from exc

        jev_bundle = JevSystemOneBundle(
            state_summary=state_str,
            choices={
                "department_category": JevChoiceResult(
                    question_key="department_category",
                    instructions=JEV_CHOICE_INSTRUCTIONS,
                    choice=chosen_category,
                    confidence=choice_conf,
                    probabilities=cat_probs,
                )
            },
            scores={
                "urgency_level": JevScoreResult(
                    question_key="urgency_level",
                    instructions=JEV_SCORE_INSTRUCTIONS,
                    score=urgency_score,
                    level=priority_level,
                    confidence=score_conf,
                    probabilities=priority_probs,
                )
            },
            nouls={
                key: JevNoulResult(
                    question_key=key,
                    instructions=JEV_NOUL_QUESTIONS[key],
                    noul=val,
                )
                for key, val in noul_map.items()
            },
        )

        return TriagePrediction(
            category=chosen_category,
            priority=priority_level,
            choice_confidence=choice_conf,
            urgency_score=urgency_score,
            category_probabilities=cat_probs,
            priority_probabilities=priority_probs,
            nouls=noul_map,
            jev_bundle=jev_bundle,
            extracted_location=extract_location(request.text, request.location),
            evidence_quotes=extract_evidence_spans(request.text),
            rationale_note=(
                f"TypeSafe Jev ({self.model}) evaluated Choice, Score, and 4 Nouls in parallel."
            ),
            backend="jev_cloud",
            model=self.model,
            calibration_status="calibrated_jev_api",
        )


def parse_and_verify_llm_jev_payload(raw_text: str, request: ResidentRequest) -> LLMJevEmulationPayload:
    """Parse LLM JSON output and verify that all evidence quotes are exact substrings of the input."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    payload = LLMJevEmulationPayload.model_validate_json(cleaned)
    verified_quotes: list[str] = []
    for quote in payload.evidence_quotes:
        candidate = quote.strip()
        if candidate not in request.text:
            stripped = candidate.strip("\"'“”‘’")
            if stripped and stripped in request.text:
                candidate = stripped
            else:
                raise ValueError("LLM evidence quote is not a verbatim span of the resident request.")
        if candidate and candidate not in verified_quotes:
            verified_quotes.append(candidate)
    return payload.model_copy(update={"evidence_quotes": verified_quotes})


class OllamaJevBackend:
    """
    Local Ollama LLM backend that emulates the Jev Choice / Score / Noul schema
    with strict Pydantic validation and verbatim evidence checking.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        chat_client: Any | None = None,
    ):
        self.model = (model or os.getenv("OLLAMA_MODEL") or "gemma3:4b").strip()
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OLLAMA_BASE_URL must be a valid http or https URL.")
        self.max_retries = max(0, min(max_retries, 3))
        self.client = chat_client or ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=0,
            format="json",
            num_ctx=4096,
            client_kwargs={"timeout": 35.0},
        )

    def predict(self, request: ResidentRequest) -> TriagePrediction:
        messages = [
            SystemMessage(content=build_ollama_jev_prompt()),
            HumanMessage(
                content=(
                    "Evaluate the following untrusted resident request state and return the Jev JSON object:\n"
                    f"<resident_state>\n{build_state_payload(request)}\n</resident_state>"
                )
            ),
        ]
        last_err: Exception | None = None
        parsed_payload: LLMJevEmulationPayload | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.invoke(messages)
            except ( httpx.HTTPError, ResponseError, ConnectionError, OSError ) as exc:
                raise BackendUnavailableError(
                    f"Could not reach local Ollama server at {self.base_url} for model '{self.model}'."
                ) from exc
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            try:
                parsed_payload = parse_and_verify_llm_jev_payload(content, request)
                break
            except (ValidationError, ValueError) as exc:
                last_err = exc
                if attempt < self.max_retries:
                    messages.append(
                        HumanMessage(
                            content=(
                                "Your previous JSON did not pass schema or verbatim-quote validation. "
                                "Return valid JSON only, and ensure evidence_quotes are exact substrings of the input."
                            )
                        )
                    )

        if parsed_payload is None:
            raise BackendUnavailableError(
                "Local Ollama model output failed Jev schema validation after retries."
            ) from last_err

        remainder = round((1.0 - parsed_payload.choice_confidence) / max(1, len(Category) - 1), 4)
        cat_probs = {
            cat.value: (
                round(parsed_payload.choice_confidence, 4)
                if cat == parsed_payload.category
                else remainder
            )
            for cat in Category
        }
        priority_probs = {
            p.value: (1.0 if p == parsed_payload.priority else 0.0) for p in Priority
        }
        noul_map = {
            "life_safety_hazard": parsed_payload.noul_life_safety_hazard,
            "multi_intent_request": parsed_payload.noul_multi_intent_request,
            "adversarial_or_sensitive": parsed_payload.noul_adversarial_or_sensitive,
            "requires_human_triage": parsed_payload.noul_requires_human_triage,
        }

        jev_bundle = JevSystemOneBundle(
            state_summary=build_state_payload(request),
            choices={
                "department_category": JevChoiceResult(
                    question_key="department_category",
                    instructions=JEV_CHOICE_INSTRUCTIONS,
                    choice=parsed_payload.category,
                    confidence=parsed_payload.choice_confidence,
                    probabilities=cat_probs,
                )
            },
            scores={
                "urgency_level": JevScoreResult(
                    question_key="urgency_level",
                    instructions=JEV_SCORE_INSTRUCTIONS,
                    score=parsed_payload.urgency_score,
                    level=parsed_payload.priority,
                    confidence=parsed_payload.choice_confidence,
                    probabilities=priority_probs,
                )
            },
            nouls={
                key: JevNoulResult(
                    question_key=key,
                    instructions=JEV_NOUL_QUESTIONS[key],
                    noul=val,
                )
                for key, val in noul_map.items()
            },
        )

        return TriagePrediction(
            category=parsed_payload.category,
            priority=parsed_payload.priority,
            choice_confidence=parsed_payload.choice_confidence,
            urgency_score=parsed_payload.urgency_score,
            category_probabilities=cat_probs,
            priority_probabilities=priority_probs,
            nouls=noul_map,
            jev_bundle=jev_bundle,
            extracted_location=extract_location(request.text, request.location),
            evidence_quotes=parsed_payload.evidence_quotes or extract_evidence_spans(request.text),
            rationale_note=parsed_payload.rationale_note,
            backend="ollama_jev",
            model=self.model,
            calibration_status="uncalibrated_llm_self_report",
        )
