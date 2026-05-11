"""
Pattern Extractor Module
========================
Provides sliding window extraction over SAX strings to construct states.
Each unique pattern extracted is defined as a distinct state in the automaton.
"""

from typing import List, Set


class PatternExtractor:
    """Extracts patterns from SAX strings to define states using a sliding window."""

    def __init__(self, window_size: int):
        """
        Args:
            window_size: The length of the sliding window (number of SAX symbols).
        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        self.window_size = window_size

    def extract_patterns(self, sax_string: str) -> List[str]:
        """
        Extracts sliding window patterns from a SAX string.

        Args:
            sax_string: The SAX representation of the time series.

        Returns:
            List of patterns (substrings of length window_size).
        """
        n = len(sax_string)
        if n < self.window_size:
            return []

        patterns = []
        for i in range(n - self.window_size + 1):
            patterns.append(sax_string[i : i + self.window_size])

        return patterns

    def get_unique_states(self, sax_string: str) -> Set[str]:
        """
        Returns the set of unique states (patterns) observed in the SAX string.

        Args:
            sax_string: The SAX representation of the time series.

        Returns:
            Set of unique states.
        """
        return set(self.extract_patterns(sax_string))
