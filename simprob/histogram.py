"""
Probability Histograms.

Useful to model state probabilities in finite state machines with the hidden_markov module.
"""

import numpy as np


class Histogram:
    "Probability histogram: An array of probabilities of discrete states"

    def __init__(self, probs: np.ndarray):
        assert (probs >= 0).all()
        assert (probs != 0).any()
        self.scaled_probs = probs / probs.max()

    @property
    def probs(self) -> np.ndarray:
        "The probabilities for each state."
        return self.scaled_probs / self.scaled_probs.sum()

    def __and__(self, other: "Histogram") -> "Histogram":
        "Fuse two distributions modelling the same random variable."
        return Histogram(self.scaled_probs * other.scaled_probs)

    @classmethod
    def empty(cls, shape):
        "Histogram representing no knowledge (all states have same probability)"
        return cls(np.ones(shape))

    def __repr__(self):
        return f"Histogram({self.probs})"
