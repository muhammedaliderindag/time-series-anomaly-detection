"""
Probabilistic Automaton Core Model
==================================
Learns and represents state transitions as a probabilistic matrix.

State transition calculation:
    P(S_i -> S_j) = Count(S_i -> S_j) / Count(S_i -> *)

The automaton is trained only on training patterns to prevent data leakage.
Unseen states during inference are mapped to the nearest known state using
Levenshtein distance with deterministic tie-breaking.
"""

import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings."""
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

        # Known states learned from training data
        self.states: Set[str] = set()

    def fit(self, patterns: List[str]) -> None:
        """
        Computes transition probabilities from a sequence of patterns.

        Args:
            patterns: Sequence of state strings extracted from training data.
        """
        if len(patterns) < 2:
            raise ValueError("At least 2 states are required to learn transitions.")

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

        self.transition_matrix = {}

        for s_from, targets in counts.items():
            self.transition_matrix[s_from] = {}

            for s_to, count in targets.items():
                self.transition_matrix[s_from][s_to] = count / totals[s_from]

    def find_nearest_state_with_distance(
        self,
        unseen_pattern: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Finds the nearest known state using Levenshtein distance.

        Deterministic tie-breaking is applied by iterating through states in
        alphabetical order.

        Args:
            unseen_pattern: Pattern that may not exist in the training states.

        Returns:
            Tuple of (nearest_state, edit_distance). If the automaton has no
            known states, returns (None, None).
        """
        if not self.states:
            return None, None

        if unseen_pattern in self.states:
            return unseen_pattern, 0

        best_state = None
        min_dist = float("inf")

        for state in sorted(self.states):
            dist = levenshtein_distance(unseen_pattern, state)

            if dist < min_dist:
                min_dist = dist
                best_state = state

        return best_state, int(min_dist)

    def find_nearest_state(self, unseen_pattern: str) -> Optional[str]:
        """
        Finds the nearest known state using Levenshtein distance.

        This method is kept for backward compatibility with existing code and
        tests. Use find_nearest_state_with_distance when the distance value is
        needed for explainability.
        """
        nearest_state, _ = self.find_nearest_state_with_distance(unseen_pattern)
        return nearest_state

    def get_transition_probability(self, s_from: str, s_to: str) -> float:
        """
        Returns the transition probability from s_from to s_to.

        If s_from is known but no transition to s_to exists, returns 0.0.
        If s_from is unknown, returns 0.0.
        """
        if s_from in self.transition_matrix:
            return self.transition_matrix[s_from].get(s_to, 0.0)

        return 0.0

    def evaluate_transition(self, s_from: str, s_to: str) -> Dict[str, Any]:
        """
        Evaluates a transition between two patterns.

        If either pattern is unseen, it is mapped to the nearest known state
        using Levenshtein distance.

        Returns:
            Dictionary containing original states, resolved states, seen/unseen
            statuses, edit distances, and transition probability.
        """
        s_from_status = "seen" if s_from in self.states else "unseen"
        s_to_status = "seen" if s_to in self.states else "unseen"

        s_from_mapped = None
        s_to_mapped = None

        s_from_distance = 0 if s_from_status == "seen" else None
        s_to_distance = 0 if s_to_status == "seen" else None

        resolved_from = s_from
        resolved_to = s_to

        if s_from_status == "unseen":
            s_from_mapped, s_from_distance = self.find_nearest_state_with_distance(s_from)
            resolved_from = s_from_mapped if s_from_mapped is not None else s_from

        if s_to_status == "unseen":
            s_to_mapped, s_to_distance = self.find_nearest_state_with_distance(s_to)
            resolved_to = s_to_mapped if s_to_mapped is not None else s_to

        probability = self.get_transition_probability(resolved_from, resolved_to)

        return {
            "original_from": s_from,
            "original_to": s_to,
            "resolved_from": resolved_from,
            "resolved_to": resolved_to,
            "s_from_status": s_from_status,
            "s_from_mapped": s_from_mapped,
            "s_from_edit_distance": s_from_distance,
            "s_to_status": s_to_status,
            "s_to_mapped": s_to_mapped,
            "s_to_edit_distance": s_to_distance,
            "transition": {
                "from": resolved_from,
                "to": resolved_to,
                "probability": probability
            },
            "probability": probability
        }

    def get_transition_count(self) -> int:
        """
        Returns the number of non-zero transitions learned by the automaton.
        Useful for transition density and parameter sensitivity analysis.
        """
        return sum(len(targets) for targets in self.transition_matrix.values())

    def get_state_count(self) -> int:
        """Returns the number of known states."""
        return len(self.states)

    def get_transition_density(self) -> float:
        """
        Returns transition density as:

            observed_transitions / possible_transitions

        where possible_transitions = number_of_states^2.
        """
        state_count = self.get_state_count()

        if state_count == 0:
            return 0.0

        possible_transitions = state_count * state_count

        if possible_transitions == 0:
            return 0.0

        return self.get_transition_count() / possible_transitions

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
            "states": sorted(list(self.states)),
            "transition_matrix": self.transition_matrix
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def load(self, filepath: str) -> None:
        """Loads a saved model from the given path."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.states = set(data.get("states", []))
        self.transition_matrix = data.get("transition_matrix", {})