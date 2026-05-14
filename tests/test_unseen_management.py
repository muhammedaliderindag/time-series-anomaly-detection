"""
Unit Tests for Unseen Pattern Management
=========================================
Tests the Levenshtein (edit distance) mapping functionality in the
ProbabilisticAutomaton class to ensure unseen patterns map to the
mathematically nearest known state.
"""

import sys
import os
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton, levenshtein_distance


class TestUnseenManagement(unittest.TestCase):
    """Tests Levenshtein mapping and distance calculations."""

    def test_levenshtein_distance(self):
        """Verifies correct calculation of edit distance between strings."""
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(levenshtein_distance("abc", "acc"), 1)
        self.assertEqual(levenshtein_distance("abc", "abd"), 1)
        self.assertEqual(levenshtein_distance("abc", "xyz"), 3)
        self.assertEqual(levenshtein_distance("abc", "ab"), 1)
        self.assertEqual(levenshtein_distance("abc", "abcd"), 1)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_nearest_state_mapping(self):
        """Proves that unseen states map correctly to the nearest known states."""
        model = ProbabilisticAutomaton()
        model.states = {"abc", "bcd", "cde"}

        # 1. Exact match
        self.assertEqual(model.find_nearest_state("abc"), "abc")

        # 2. Nearest match (Distance 1)
        self.assertEqual(model.find_nearest_state("acc"), "abc")

        # 3. Tie-breaking check (bce is dist 1 from bcd and cde. Alphabetical order ensures bcd is picked)
        self.assertEqual(model.find_nearest_state("bce"), "bcd")

        # 4. Tie-breaking check (xyz is dist 3 from all. Alphabetical order ensures abc is picked)
        self.assertEqual(model.find_nearest_state("xyz"), "abc")

    def test_nearest_state_empty_model(self):
        """Ensures None is returned if no states have been learned."""
        model = ProbabilisticAutomaton()
        self.assertIsNone(model.find_nearest_state("abc"))


if __name__ == "__main__":
    unittest.main()
