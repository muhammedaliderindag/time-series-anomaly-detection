"""
Probabilistic Automaton Core Model
==================================
Learns and represents state transitions as a probabilistic matrix.
State transition calculation: P(S_i -> S_j) = Count(S_i -> S_j) / Count(S_i -> *)
Only trained on training data to prevent leakage. Supports persistence.
Handles unseen states during inference using Levenshtein distance mapping.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class ProbabilisticAutomaton:
    """Core Probabilistic Automaton model for time-series anomaly detection."""

    def __init__(self, model_dir: str = "./models"):
        """
        Args:
            model_dir: Base directory for storing model artifacts.
        """
        self.model_dir = model_dir
        self.artifact_dir = os.path.join(model_dir, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

        # transition_matrix[state_from][state_to] = probability
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        # Keep track of known states
        self.states: Set[str] = set()

    def fit(self, patterns: List[str]) -> None:
        """
        Computes transition probabilities from a sequence of patterns (states).

        Args:
            patterns: Sequence of state strings extracted from training data.
        """
        if len(patterns) < 2:
            raise ValueError("At least 2 states are required to learn transitions.")

        # Count state-to-state transitions
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)

        self.states.clear()
        for i in range(len(patterns) - 1):
            s_from = patterns[i]
            s_to = patterns[i + 1]

            counts[s_from][s_to] += 1
            totals[s_from] += 1
            self.states.add(s_from)
            self.states.add(s_to)

        # Calculate transition probabilities
        self.transition_matrix = {}
        for s_from, targets in counts.items():
            self.transition_matrix[s_from] = {}
            for s_to, count in targets.items():
                self.transition_matrix[s_from][s_to] = count / totals[s_from]

    def find_nearest_state(self, unseen_pattern: str) -> Optional[str]:
        """
        Finds the nearest known state in self.states using Levenshtein distance.
        Deterministic tie-breaking using alphabetical order.
        """
        if not self.states:
            return None
        if unseen_pattern in self.states:
            return unseen_pattern

        best_state = None
        min_dist = float("inf")

        for state in sorted(self.states):
            dist = levenshtein_distance(unseen_pattern, state)
            if dist < min_dist:
                min_dist = dist
                best_state = state

        return best_state

    def get_transition_probability(self, s_from: str, s_to: str) -> float:
        """
        Returns the transition probability from s_from to s_to.
        If s_from is known but no transition to s_to exists, returns 0.0.
        If s_from is unknown, returns 0.0.
        """
        if s_from in self.transition_matrix:
            return self.transition_matrix[s_from].get(s_to, 0.0)
        return 0.0

    def evaluate_transition(self, s_from: str, s_to: str) -> Dict[str, any]:
        """
        Evaluates a transition between two patterns. If either is unseen,
        maps them to the nearest known states using Levenshtein distance.

        Returns:
            Dictionary containing evaluation details:
            {
                "s_from_status": "seen" | "unseen",
                "s_from_mapped": str | None,
                "s_to_status": "seen" | "unseen",
                "s_to_mapped": str | None,
                "probability": float
            }
        """
        s_from_status = "seen" if s_from in self.states else "unseen"
        s_to_status = "seen" if s_to in self.states else "unseen"

        s_from_mapped = None
        s_to_mapped = None

        resolved_from = s_from
        resolved_to = s_to

        if s_from_status == "unseen":
            s_from_mapped = self.find_nearest_state(s_from)
            resolved_from = s_from_mapped if s_from_mapped is not None else s_from

        if s_to_status == "unseen":
            s_to_mapped = self.find_nearest_state(s_to)
            resolved_to = s_to_mapped if s_to_mapped is not None else s_to

        prob = self.get_transition_probability(resolved_from, resolved_to)

        return {
            "s_from_status": s_from_status,
            "s_from_mapped": s_from_mapped,
            "s_to_status": s_to_status,
            "s_to_mapped": s_to_mapped,
            "probability": prob
        }

    def save(self, filename: str = "automaton_model.json") -> str:
        """
        Saves the learned transition matrix and state set.

        Args:
            filename: Target file name.

        Returns:
            The filepath where the model was saved.
        """
        filepath = os.path.join(self.artifact_dir, filename)
        data = {
            "states": list(self.states),
            "transition_matrix": self.transition_matrix
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def load(self, filepath: str) -> None:
        """
        Loads a saved model from the given path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.states = set(data.get("states", []))
        self.transition_matrix = data.get("transition_matrix", {})
