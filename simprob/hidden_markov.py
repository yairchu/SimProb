"""
Finite state machine models (aka Hidden Markov models).

This module defines transition classes for use with the smoothing module.

See https://en.wikipedia.org/wiki/Hidden_Markov_model
"""

import dataclasses
import numpy as np
import scipy

from .histogram import Histogram


@dataclasses.dataclass
class TransitionMatrix:
    mat: np.ndarray

    def __call__(self, state: Histogram) -> Histogram:
        return type(state)(self.mat @ state.scaled_probs)

    @property
    def inv(self) -> "TransitionMatrix":
        return TransitionMatrix(self.mat.T)


@dataclasses.dataclass
class ConvolutionTransition:
    kernel: np.ndarray

    def __call__(self, state: Histogram) -> Histogram:
        return type(state)(
            scipy.signal.convolve(state.scaled_probs, self.kernel, mode="same")
        )

    @property
    def inv(self) -> "ConvolutionTransition":
        inverse_kernel = self.kernel[(slice(None, None, -1),) * len(self.kernel.shape)]
        return type(self)(inverse_kernel)
