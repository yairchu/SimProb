"""
Transition class for Kalman Filtering.

Intended to model transitions between multivar_normal.MultivariateNormal states
using the simulation.simulate function.

See https://en.wikipedia.org/wiki/Kalman_filter
"""

import dataclasses
import functools
import numpy as np
import operator

from .multivar_normal import MultivariateNormal
from .subspace_metric import SubspaceMetric


@dataclasses.dataclass
class KalmanTransition:
    transition_matrix: np.ndarray  # How state evolves (F)

    def __post_init__(self):
        # Translate lists to numpy arrays
        self.transition_matrix = np.asarray(self.transition_matrix)

    def __call__(self, state):
        return self.transition_matrix @ state

    def inv(self, state):
        return np.linalg.inv(self.transition_matrix) @ state


def add_process_noise(q, means=None):
    q = np.asarray(q)
    if means is None:
        means = np.zeros(q.shape[0])
    return functools.partial(operator.add, MultivariateNormal(mean=means, covar=q))
