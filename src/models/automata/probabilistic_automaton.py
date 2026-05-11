"""
Probabilistic Automaton Core Model
==================================
Learns and represents state transitions as a probabilistic matrix.
State transition calculation: P(S_i -> S_j) = Count(S_i -> S_j) / Count(S_i -> *)
Only trained on training data to prevent leakage. Supports persistence.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Optional


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

    def get_transition_probability(self, s_from: str, s_to: str) -> float:
        """
        Returns the transition probability from s_from to s_to.
        If s_from is known but no transition to s_to exists, returns 0.0.
        If s_from is unknown, returns 0.0 (or needs unseen handling).
        """
        if s_from in self.transition_matrix:
            return self.transition_matrix[s_from].get(s_to, 0.0)
        return 0.0

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
