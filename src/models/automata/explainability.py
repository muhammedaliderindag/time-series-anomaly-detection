"""
Probabilistic Explainability Module
===================================
Evaluates path transitions, computes overall path probabilities, and provides
step-by-step decision justification with strict JSON-ready output formatting.

The explanation is based directly on the probabilistic automaton internals:
transition probabilities, unseen pattern mapping, and path probability.
"""

import json
from typing import Any, Dict, List, Tuple

from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton


class AutomataExplainability:
    """Provides path evaluation and JSON-formatted justifications for automaton decisions."""

    def __init__(self, model: ProbabilisticAutomaton, anomaly_threshold: float):
        """
        Args:
            model: Trained ProbabilisticAutomaton model.
            anomaly_threshold: Probability threshold below which a transition is considered anomalous.
        """
        self.model = model
        self.anomaly_threshold = anomaly_threshold

    def _make_reason(self, probability: float, decision: str) -> str:
        """Creates a deterministic probabilistic explanation for a decision."""
        if decision == "anomaly":
            return (
                f"Transition probability {probability:.6f} is below "
                f"the anomaly threshold {self.anomaly_threshold:.6f}."
            )

        return (
            f"Transition probability {probability:.6f} is greater than or equal to "
            f"the anomaly threshold {self.anomaly_threshold:.6f}."
        )

    def explain_path(self, raw_patterns: List[str]) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Evaluates a sequence of observed patterns, tracking transitions and anomalies.

        Args:
            raw_patterns: List of SAX patterns/states observed during inference.

        Returns:
            Tuple of:
            - List of step-by-step JSON-ready justification dictionaries.
            - Overall path probability as the product of transition probabilities.
            - Confidence score as the mean transition probability.
        """
        if not raw_patterns:
            return [], 1.0, 1.0

        justifications: List[Dict[str, Any]] = []
        path_probability = 1.0
        transition_probabilities: List[float] = []

        first_pattern = raw_patterns[0]
        first_state, first_distance = self.model.find_nearest_state_with_distance(first_pattern)
        first_status = "seen" if first_pattern in self.model.states else "unseen"

        first_mapped = None if first_status == "seen" else first_state
        first_resolved = first_pattern if first_status == "seen" else first_state

        if first_resolved is None:
            first_resolved = first_pattern

        justifications.append({
            "time_step": 0,
            "previous_state": None,
            "state": first_resolved,
            "pattern": first_pattern,
            "status": first_status,
            "mapped_to": first_mapped,
            "edit_distance": first_distance,
            "transition": None,
            "probability": 1.0,
            "path_probability_so_far": 1.0,
            "threshold": float(self.anomaly_threshold),
            "decision": "normal",
            "reason": "Initial state has no incoming transition."
        })

        current_resolved = first_resolved

        for t in range(1, len(raw_patterns)):
            next_pattern = raw_patterns[t]
            eval_res = self.model.evaluate_transition(current_resolved, next_pattern)

            probability = float(eval_res["probability"])
            path_probability *= probability
            transition_probabilities.append(probability)

            decision = "normal" if probability >= self.anomaly_threshold else "anomaly"
            reason = self._make_reason(probability, decision)

            next_status = eval_res["s_to_status"]
            next_mapped = eval_res["s_to_mapped"]
            next_resolved = eval_res["resolved_to"]

            justifications.append({
                "time_step": t,
                "previous_state": eval_res["resolved_from"],
                "state": next_resolved,
                "pattern": next_pattern,
                "status": next_status,
                "mapped_to": next_mapped,
                "edit_distance": eval_res["s_to_edit_distance"],
                "transition": eval_res["transition"],
                "probability": probability,
                "path_probability_so_far": float(path_probability),
                "threshold": float(self.anomaly_threshold),
                "decision": decision,
                "reason": reason
            })

            current_resolved = next_resolved

        confidence_score = (
            float(sum(transition_probabilities) / len(transition_probabilities))
            if transition_probabilities
            else 1.0
        )

        return justifications, float(path_probability), confidence_score

    def build_summary(
        self,
        justifications: List[Dict[str, Any]],
        path_probability: float,
        confidence_score: float
    ) -> Dict[str, Any]:
        """
        Builds a compact summary for report-ready JSON output.

        path_probability keeps the mathematically required product of transition
        probabilities. confidence_score is reported as the mean transition
        probability to avoid long paths collapsing interpretability to zero.
        """
        transition_probabilities = [
            float(step["probability"])
            for step in justifications
            if step.get("transition") is not None
        ]

        final_decision = (
            "anomaly"
            if any(step.get("decision") == "anomaly" for step in justifications)
            else "normal"
        )

        mean_transition_probability = (
            float(sum(transition_probabilities) / len(transition_probabilities))
            if transition_probabilities
            else 1.0
        )

        min_transition_probability = (
            float(min(transition_probabilities))
            if transition_probabilities
            else 1.0
        )

        zero_probability_transitions = sum(
            1 for prob in transition_probabilities if prob == 0.0
        )

        anomalous_transition_count = sum(
            1 for step in justifications if step.get("decision") == "anomaly"
        )

        transition_count = len(transition_probabilities)

        return {
            "path_probability": float(path_probability),
            "confidence_score": float(confidence_score),
            "mean_transition_probability": mean_transition_probability,
            "min_transition_probability": min_transition_probability,
            "zero_probability_transitions": zero_probability_transitions,
            "final_decision": final_decision,
            "anomaly_threshold": float(self.anomaly_threshold),
            "num_steps": len(justifications),
            "num_transitions": transition_count,
            "num_anomalous_transitions": anomalous_transition_count,
            "anomalous_transition_rate": (
                float(anomalous_transition_count / transition_count)
                if transition_count > 0
                else 0.0
            ),
            "reason": (
                "At least one low-probability transition was detected."
                if final_decision == "anomaly"
                else "All observed transitions were above the anomaly threshold."
            )
        }

    def format_json_output(self, justifications: List[Dict[str, Any]]) -> str:
        """Converts justification dictionaries to a formatted JSON string."""
        return json.dumps(justifications, indent=2, ensure_ascii=False)