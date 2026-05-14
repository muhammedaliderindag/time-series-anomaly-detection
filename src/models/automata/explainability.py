"""
Probabilistic Explainability Module
===================================
Evaluates path transitions, computes overall path probabilities, and provides
step-by-step decision justification with strict JSON output formatting.
"""

import json
from typing import List, Dict, Tuple, Any
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton


class AutomataExplainability:
    """Provides path evaluation and JSON-formatted justifications for automaton decisions."""

    def __init__(self, model: ProbabilisticAutomaton, anomaly_threshold: float):
        """
        Args:
            model: Trained ProbabilisticAutomaton model.
            anomaly_threshold: Transition probability threshold below which a state is an anomaly.
        """
        self.model = model
        self.anomaly_threshold = anomaly_threshold

    def explain_path(self, raw_patterns: List[str]) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Evaluates a sequence of observed patterns, tracking transitions and anomalies.

        Args:
            raw_patterns: List of SAX patterns (states) observed in inference.

        Returns:
            Tuple of:
            - List of step-by-step justification dicts in strict JSON-ready format.
            - Overall path probability (product of transition probabilities).
            - Confidence score (average probability of the transitions).
        """
        if not raw_patterns:
            return [], 1.0, 1.0

        justifications = []
        path_probability = 1.0
        probabilities_list = []

        # Step 0: Initial state (no transition yet)
        first_pattern = raw_patterns[0]
        first_status = "seen" if first_pattern in self.model.states else "unseen"
        first_mapped = None if first_status == "seen" else self.model.find_nearest_state(first_pattern)
        first_resolved = first_pattern if first_status == "seen" else first_mapped

        justifications.append({
            "time_step": 0,
            "state": first_resolved,
            "pattern": first_pattern,
            "status": first_status,
            "mapped_to": first_mapped,
            "probability": 1.0,
            "decision": "normal"
        })

        current_resolved = first_resolved

        # Steps 1 to N: Evaluate transitions
        for t in range(1, len(raw_patterns)):
            next_pattern = raw_patterns[t]

            # Evaluate transition from current_resolved to next_pattern
            eval_res = self.model.evaluate_transition(current_resolved, next_pattern)

            prob = eval_res["probability"]
            path_probability *= prob
            probabilities_list.append(prob)

            # Determine decision based on threshold
            decision = "normal" if prob >= self.anomaly_threshold else "anomaly"

            # The next state in the sequence becomes the resolved target state for the next step
            next_status = eval_res["s_to_status"]
            next_mapped = eval_res["s_to_mapped"]
            next_resolved = next_pattern if next_status == "seen" else next_mapped

            justifications.append({
                "time_step": t,
                "state": next_resolved,
                "pattern": next_pattern,
                "status": next_status,
                "mapped_to": next_mapped,
                "probability": float(prob),
                "decision": decision
            })

            current_resolved = next_resolved

        # Calculate confidence score (average of transition probabilities)
        confidence_score = sum(probabilities_list) / len(probabilities_list) if probabilities_list else 1.0

        return justifications, float(path_probability), float(confidence_score)

    def format_json_output(self, justifications: List[Dict[str, Any]]) -> str:
        """Utility to convert justification dicts to a formatted JSON string."""
        return json.dumps(justifications, indent=2, ensure_ascii=False)
