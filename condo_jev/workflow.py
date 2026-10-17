from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from condo_jev.backends import BackendUnavailableError, TriageBackend, build_state_payload
from condo_jev.schema import (
    Category,
    Classification,
    JevChoiceResult,
    JevNoulResult,
    JevScoreResult,
    JevSystemOneBundle,
    Priority,
    ResidentRequest,
    TriagePrediction,
)
from condo_jev.signals import (
    SafetyScreenSignal,
    extract_location,
    resolve_domain_category_override,
    screen_request_safety,
)
from condo_jev.taxonomy import (
    JEV_CHOICE_INSTRUCTIONS,
    JEV_NOUL_QUESTIONS,
    JEV_SCORE_INSTRUCTIONS,
    TAXONOMY,
)


class TriageGraphState(TypedDict, total=False):
    request: ResidentRequest
    safety_signal: SafetyScreenSignal | None
    prediction: TriagePrediction | None
    backend_error: str | None
    classification: Classification


class JevTriageService:
    """
    LangGraph-orchestrated Condominium Resident Request Triage service built around
    Jev's System-One primitives (`Choice`, `Score`, and `Noul`).
    """

    def __init__(
        self,
        backend: TriageBackend,
        confidence_threshold: float = 0.55,
        noul_review_threshold: float = 0.50,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= noul_review_threshold <= 1.0:
            raise ValueError("noul_review_threshold must be between 0.0 and 1.0.")

        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.noul_review_threshold = noul_review_threshold

        builder = StateGraph(TriageGraphState)
        builder.add_node("safety_gate", self._node_safety_gate)
        builder.add_node("jev_system_one", self._node_jev_system_one)
        builder.add_node("jev_policy_router", self._node_jev_policy_router)

        builder.add_edge(START, "safety_gate")
        builder.add_conditional_edges(
            "safety_gate",
            lambda state: "jev_policy_router" if state.get("safety_signal") is not None else "jev_system_one",
            ["jev_system_one", "jev_policy_router"],
        )
        builder.add_edge("jev_system_one", "jev_policy_router")
        builder.add_edge("jev_policy_router", END)
        self.graph = builder.compile()

    def classify(self, request: ResidentRequest) -> Classification:
        t0 = perf_counter()
        final_state = self.graph.invoke(
            {
                "request": request,
                "safety_signal": None,
                "prediction": None,
                "backend_error": None,
            }
        )
        elapsed_ms = round((perf_counter() - t0) * 1000.0, 2)
        result: Classification = final_state["classification"]
        return result.model_copy(update={"latency_ms": elapsed_ms})

    def _node_safety_gate(self, state: TriageGraphState) -> dict[str, SafetyScreenSignal | None]:
        req = state["request"]
        return {"safety_signal": screen_request_safety(req.text)}

    def _node_jev_system_one(self, state: TriageGraphState) -> dict[str, TriagePrediction | str | None]:
        req = state["request"]
        try:
            prediction = self.backend.predict(req)
            return {"prediction": prediction, "backend_error": None}
        except BackendUnavailableError as exc:
            return {"prediction": None, "backend_error": str(exc)}

    def _node_jev_policy_router(self, state: TriageGraphState) -> dict[str, Classification]:
        req = state["request"]
        signal = state.get("safety_signal")
        prediction = state.get("prediction")
        backend_error = state.get("backend_error")

        if signal is not None:
            return {"classification": self._build_safety_classification(req, signal)}

        if prediction is None:
            return {"classification": self._build_fallback_classification(req, backend_error or "Backend unavailable.")}

        reasons: list[str] = []

        if prediction.category == Category.OTHER:
            reasons.append("Jev Choice selected 'other' (insufficient or out-of-scope request context).")

        if prediction.choice_confidence < self.confidence_threshold:
            reasons.append(
                f"Jev Choice concentration confidence ({prediction.choice_confidence:.0%}) is below policy threshold ({self.confidence_threshold:.0%})."
            )

        noul_hazard = prediction.nouls.get("life_safety_hazard", 0.0)
        noul_multi = prediction.nouls.get("multi_intent_request", 0.0)
        noul_adv = prediction.nouls.get("adversarial_or_sensitive", 0.0)
        noul_review = prediction.nouls.get("requires_human_triage", 0.0)

        if noul_hazard >= self.noul_review_threshold or prediction.priority == Priority.EMERGENCY:
            reasons.append(
                f"Jev Noul 'life_safety_hazard' is elevated (P={noul_hazard:.2f}) or priority is emergency."
            )

        if noul_multi >= self.noul_review_threshold:
            reasons.append(
                f"Jev Noul 'multi_intent_request' indicates multiple independent topics (P={noul_multi:.2f})."
            )

        if noul_adv >= self.noul_review_threshold:
            reasons.append(
                f"Jev Noul 'adversarial_or_sensitive' flagged policy/escalation risk (P={noul_adv:.2f})."
            )

        if noul_review >= self.noul_review_threshold and not reasons:
            reasons.append(
                f"Jev Noul 'requires_human_triage' (P={noul_review:.2f}) reached review threshold ({self.noul_review_threshold:.2f})."
            )

        if prediction.calibration_status == "uncalibrated_llm_self_report":
            reasons.append("Local LLM self-reported Jev scores are uncalibrated and require staff verification.")

        final_category = resolve_domain_category_override(req.text, prediction.category)
        spec = TAXONOMY[final_category]
        missing_details: list[str] = []
        if spec.needs_location and not prediction.extracted_location:
            missing_details.append("Provide the block, floor, unit or common-area location.")

        return {
            "classification": Classification(
                category=final_category,
                category_title=spec.title,
                assigned_team=spec.team,
                sla_target=spec.sla_window,
                priority=prediction.priority,
                choice_confidence=prediction.choice_confidence,
                urgency_score=prediction.urgency_score,
                calibration_status=prediction.calibration_status,
                category_probabilities=prediction.category_probabilities,
                priority_probabilities=prediction.priority_probabilities,
                nouls=prediction.nouls,
                jev_bundle=(
                    prediction.jev_bundle.model_dump()
                    if hasattr(prediction.jev_bundle, "model_dump")
                    else prediction.jev_bundle
                ),
                needs_human_review=bool(reasons),
                review_reasons=reasons,
                missing_details=missing_details,
                extracted_location=prediction.extracted_location,
                evidence_quotes=prediction.evidence_quotes,
                rationale_note=prediction.rationale_note,
                backend=prediction.backend,
                model=prediction.model,
                latency_ms=0.0,
            )
        }

    def _build_safety_classification(
        self,
        request: ResidentRequest,
        signal: SafetyScreenSignal,
    ) -> Classification:
        spec = TAXONOMY[signal.category]
        base_cat_probs: dict[str, float] = {}
        base_prio_probs: dict[str, float] = {}
        try:
            if isinstance(self.backend, LocalJevSimulatorBackend):
                sim_pred = self.backend.predict(request)
                base_cat_probs = sim_pred.category_probabilities
                base_prio_probs = sim_pred.priority_probabilities
        except Exception:
            pass

        if base_cat_probs:
            raw_cat = {
                cat.value: 0.78 * (1.0 if cat == signal.category else 0.0)
                + 0.22 * float(base_cat_probs.get(cat.value, 0.01))
                for cat in Category
            }
        else:
            raw_cat = {
                cat.value: (0.865 if cat == signal.category else 0.015)
                for cat in Category
            }
        cat_total = sum(raw_cat.values()) or 1.0
        cat_probs = {k: round(v / cat_total, 4) for k, v in raw_cat.items()}

        if base_prio_probs:
            raw_prio = {
                p.value: 0.80 * (1.0 if p == signal.priority else 0.0)
                + 0.20 * float(base_prio_probs.get(p.value, 0.02))
                for p in Priority
            }
        else:
            raw_prio = {
                p.value: (0.88 if p == signal.priority else 0.04)
                for p in Priority
            }
        prio_total = sum(raw_prio.values()) or 1.0
        prio_probs = {k: round(v / prio_total, 4) for k, v in raw_prio.items()}

        choice_conf = round(cat_probs.get(signal.category.value, 0.90), 4)
        urgency_val = 0.98 if signal.priority == Priority.EMERGENCY else 0.82
        nouls = {
            "life_safety_hazard": 0.99 if signal.kind == "emergency_hazard" else 0.06,
            "multi_intent_request": 0.08,
            "adversarial_or_sensitive": 0.99 if signal.kind in {"prompt_injection", "sensitive_escalation"} else 0.04,
            "requires_human_triage": 0.99,
        }
        bundle = JevSystemOneBundle(
            state_summary=build_state_payload(request),
            choices={
                "department_category": JevChoiceResult(
                    question_key="department_category",
                    instructions=JEV_CHOICE_INSTRUCTIONS,
                    choice=signal.category,
                    confidence=choice_conf,
                    probabilities=cat_probs,
                )
            },
            scores={
                "urgency_level": JevScoreResult(
                    question_key="urgency_level",
                    instructions=JEV_SCORE_INSTRUCTIONS,
                    score=urgency_val,
                    level=signal.priority,
                    confidence=round(prio_probs.get(signal.priority.value, 0.90), 4),
                    probabilities=prio_probs,
                )
            },
            nouls={
                key: JevNoulResult(
                    question_key=key,
                    instructions=JEV_NOUL_QUESTIONS[key],
                    noul=val,
                )
                for key, val in nouls.items()
            },
        )
        return Classification(
            category=signal.category,
            category_title=spec.title,
            assigned_team=spec.team,
            sla_target=spec.sla_window,
            priority=signal.priority,
            choice_confidence=choice_conf,
            urgency_score=urgency_val,
            calibration_status="deterministic_safety_rule",
            category_probabilities=cat_probs,
            priority_probabilities=prio_probs,
            nouls=nouls,
            jev_bundle=bundle.model_dump(),
            needs_human_review=True,
            review_reasons=[signal.reason],
            extracted_location=extract_location(request.text, request.location),
            evidence_quotes=[signal.matched_span],
            rationale_note=signal.reason,
            backend="safety_gate",
            model="deterministic-guardrail-v1",
            latency_ms=0.0,
        )

    def _build_fallback_classification(
        self,
        request: ResidentRequest,
        error_message: str,
    ) -> Classification:
        spec = TAXONOMY[Category.OTHER]
        uniform = round(1.0 / len(Category), 4)
        cat_probs = {cat.value: uniform for cat in Category}
        prio_probs = {
            Priority.LOW.value: 0.0,
            Priority.NORMAL.value: 0.0,
            Priority.HIGH.value: 1.0,
            Priority.EMERGENCY.value: 0.0,
        }
        nouls = {
            "life_safety_hazard": 0.10,
            "multi_intent_request": 0.10,
            "adversarial_or_sensitive": 0.10,
            "requires_human_triage": 1.0,
        }
        bundle = JevSystemOneBundle(
            state_summary=build_state_payload(request),
            choices={
                "department_category": JevChoiceResult(
                    question_key="department_category",
                    instructions=JEV_CHOICE_INSTRUCTIONS,
                    choice=Category.OTHER,
                    confidence=0.0,
                    probabilities=cat_probs,
                )
            },
            scores={
                "urgency_level": JevScoreResult(
                    question_key="urgency_level",
                    instructions=JEV_SCORE_INSTRUCTIONS,
                    score=0.70,
                    level=Priority.HIGH,
                    confidence=0.0,
                    probabilities=prio_probs,
                )
            },
            nouls={
                key: JevNoulResult(
                    question_key=key,
                    instructions=JEV_NOUL_QUESTIONS[key],
                    noul=val,
                )
                for key, val in nouls.items()
            },
        )
        return Classification(
            category=Category.OTHER,
            category_title=spec.title,
            assigned_team=spec.team,
            sla_target=spec.sla_window,
            priority=Priority.HIGH,
            choice_confidence=0.0,
            urgency_score=0.70,
            calibration_status="deterministic_safety_rule",
            category_probabilities=cat_probs,
            priority_probabilities=prio_probs,
            nouls=nouls,
            jev_bundle=bundle.model_dump(),
            needs_human_review=True,
            review_reasons=[f"Backend fallback triggered: {error_message}"],
            extracted_location=extract_location(request.text, request.location),
            evidence_quotes=[],
            rationale_note=error_message,
            backend="fallback",
            model="unavailable",
            latency_ms=0.0,
        )


def build_follow_up_and_dispatch(
    request: ResidentRequest,
    result: Classification,
) -> dict[str, str]:
    """
    Generate a ready-to-send Resident Acknowledgement Reply and an Internal Estate
    Team Dispatch Brief based on Jev System-One Choice, Score, and Noul outputs.
    """
    loc_label = result.extracted_location or request.location or "Location not yet specified"
    noul_hazard = result.nouls.get("life_safety_hazard", 0.0)
    noul_multi = result.nouls.get("multi_intent_request", 0.0)

    from condo_jev.ai_response import generate_ai_resident_reply

    resident_reply = generate_ai_resident_reply(request, result)
    if noul_multi >= 0.50:
        resident_reply += (
            f"\n\n*(Note: Your note covers more than one topic, so we're splitting the ticket to attend to each part promptly.)*"
        )

    quotes_str = ", ".join(f'"{q}"' for q in result.evidence_quotes) if result.evidence_quotes else "None extracted"
    review_str = (
        "MANUAL REVIEW REQUIRED — " + " | ".join(result.review_reasons)
        if result.needs_human_review
        else "AUTO-DISPATCH CLEARED (All Jev Noul & Confidence Gates Passed)"
    )

    dispatch_brief = (
        f"[JEV SYSTEM-ONE DISPATCH BRIEF]\n"
        f"• Assigned Team : {result.assigned_team} ({result.category_title})\n"
        f"• Priority / SLA: {result.priority.value.upper()} (Score: {result.urgency_score:.0%}) · {result.sla_target}\n"
        f"• Location      : {loc_label} (Channel: {request.channel})\n"
        f"• Choice Conf.  : {result.choice_confidence:.0%} ({result.calibration_status})\n"
        f"• Evidence Spans: {quotes_str}\n"
        f"• Routing Gate  : {review_str}"
    )

    return {
        "resident_reply": resident_reply,
        "dispatch_brief": dispatch_brief,
    }

