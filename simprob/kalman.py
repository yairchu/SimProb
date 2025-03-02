"""
Transition class for Kalman Filtering.

Intended to model transitions between multivar_normal.MultivariateNormal states
using the simulation.simulate function.

See https://en.wikipedia.org/wiki/Kalman_filter
"""

import dataclasses
import numpy as np
import typing

from .multivar_normal import MultivariateNormal


class KalmanState(typing.Protocol):
    def __add__(self, other: "KalmanState") -> "KalmanState": ...
    def __sub__(self, other: "KalmanState") -> "KalmanState": ...
    def __rmatmul__(self, other: np.ndarray) -> "KalmanState": ...


@dataclasses.dataclass
class KalmanTransition:
    transition_matrix: np.ndarray  # How state evolves (F)
    expected_state_change: KalmanState  # Expected addition to state (B@u)

    def __call__(self, state: KalmanState) -> KalmanState:
        return self.transition_matrix @ state + self.expected_state_change

    def inv(self, state: KalmanState) -> KalmanState:
        return np.linalg.inv(self.transition_matrix) @ (
            state - self.expected_state_change
        )
