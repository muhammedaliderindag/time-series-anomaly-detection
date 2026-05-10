"""
PAA and SAX Transformations Module
===================================
Provides Piecewise Aggregate Approximation (PAA) and Symbolic Aggregate
approximation (SAX) for z-normalized time-series dimensionality reduction
and discretization.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Union


def z_normalize(data: np.ndarray) -> np.ndarray:
    """Z-normalizes a z-score of the data to have mean=0 and std=1."""
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - np.mean(data)) / std


def piecewise_aggregate_approximation(data: np.ndarray, segment_size: int) -> np.ndarray:
    """
    Reduces the z-score of the time series by averaging non-overlapping segments.

    Args:
        data: 1D numpy array representing the time series.
        segment_size: Number of data points to average per segment.

    Returns:
        PAA-reduced 1D numpy array.
    """
    n = len(data)
    if segment_size <= 1:
        return data.copy()

    # Truncate to make divisible, or pad. Truncating is simple and clean for long time series.
    num_segments = n // segment_size
    if num_segments == 0:
        return np.array([np.mean(data)])

    truncated_data = data[:num_segments * segment_size]
    reshaped = truncated_data.reshape(num_segments, segment_size)
    return np.mean(reshaped, axis=1)


def get_sax_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Computes Gaussian breakpoints partition for a z-score of the standard normal distribution.

    Args:
        alphabet_size: Size of the SAX alphabet.

    Returns:
        1D array of breakpoints of size alphabet_size - 1.
    """
    return stats.norm.ppf(np.arange(1, alphabet_size) / alphabet_size)


def symbolic_aggregate_approximation(paa_data: np.ndarray, alphabet_size: int) -> str:
    """
    Converts PAA z-score values into a symbolic SAX string.

    Args:
        paa_data: 1D numpy array of PAA-reduced values.
        alphabet_size: Number of characters in the SAX alphabet (max 26).

    Returns:
        SAX representation as a string of lowercase letters.
    """
    if alphabet_size < 2 or alphabet_size > 26:
        raise ValueError("Alphabet size must be between 2 and 26.")

    # Standard z-normalization is required before SAX z-score mapping
    normalized_paa = z_normalize(paa_data)
    breakpoints = get_sax_breakpoints(alphabet_size)

    # Map each value to a character
    alphabet = [chr(97 + i) for i in range(alphabet_size)]  # 'a', 'b', 'c', ...
    sax_chars = []

    for val in normalized_paa:
        # Find the index of the first breakpoint that is larger than val
        idx = np.searchsorted(breakpoints, val)
        sax_chars.append(alphabet[idx])

    return "".join(sax_chars)


class SAXTransformer:
    """Class wrapper for PAA and SAX transforms to keep state and configuration."""

    def __init__(self, segment_size: int, alphabet_size: int):
        self.segment_size = segment_size
        self.alphabet_size = alphabet_size

    def transform(self, data: Union[np.ndarray, List[float]]) -> str:
        """Runs PAA and then SAX to return a symbolic string representation."""
        arr = np.asarray(data, dtype=float).flatten()
        paa = piecewise_aggregate_approximation(arr, self.segment_size)
        return symbolic_aggregate_approximation(paa, self.alphabet_size)
