"""
Multivariate normal (Gaussian) distributions.

Unlike scipy.stats.multivariate_normal, MultivariateNormal implements many useful methods.

Also supports uniform distributions by using PartialCovar as its covariance matrix.
"""

import dataclasses
import numpy as np
import typing

from .partial_covar import PartialCovar


@dataclasses.dataclass
class MultivariateNormal:
    "A multivariate normal (Gaussian) distribution."

    mean: np.ndarray  # Mean vector
    covar: typing.Union[np.ndarray, PartialCovar]  # Covariance matrix

    def __post_init__(self):
        assert self.covar.shape == (self.dim, self.dim), (
            "Covariance matrix shape mismatch."
        )
        self.covar = PartialCovar.assure_symmetric_matrix(self.covar)

    @property
    def dim(self) -> int:
        "The number of dimensions of the distribution."
        return self.mean.shape[0]

    def __add__(self, other) -> "MultivariateNormal":
        "Compute the distribution of the sum of two independent Gaussian-distributed variables."
        if isinstance(other, np.ndarray):
            other = type(self).delta(other)
        if other.dim < self.dim:
            other = other.extend(self.dim)
        elif self.dim < other.dim:
            return other + self
        return type(self)(mean=self.mean + other.mean, covar=self.covar + other.covar)

    def __sub__(self, other) -> "MultivariateNormal":
        "Compute  the distribution of the difference of two independent Gaussian-distributed variables."
        return self + (-other)

    def __neg__(self) -> "MultivariateNormal":
        "The distribution of the negation of a Gaussian-distributed variable."
        return type(self)(mean=-self.mean, covar=self.covar)

    __array_priority__ = 1000  # Ensures NumPy prefers calling our `__rmatmul__`

    def __rmatmul__(self, other: np.ndarray) -> "MultivariateNormal":
        "Apply a linear transformation to the distribution."
        if self.dim < other.shape[1]:
            return other @ self.extend(other.shape[1])
        return type(self)(
            mean=other @ self.mean, covar=PartialCovar.transform(self.covar, other)
        )

    def __and__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Fuse two Gaussian distributions modelling the same random variable."
        if other.dim < self.dim:
            other = other.extend(self.dim)
        elif self.dim < other.dim:
            return other & self

        self_inv = PartialCovar.inv(self.covar)
        other_inv = PartialCovar.inv(other.covar)
        covar = PartialCovar.inv(self_inv + other_inv)
        return type(self)(
            mean=covar.real @ (self_inv @ self.mean + other_inv @ other.mean),
            covar=covar,
        )

    def extend(self, dim: int) -> "MultivariateNormal":
        add = dim - self.dim
        return type(self)(
            np.pad(self.mean, (0, add)), PartialCovar.extend(self.covar, dim)
        )

    @classmethod
    def delta(cls, point: np.ndarray) -> "MultivariateNormal":
        "A point distribution (value is completely known)"
        [n] = point.shape
        return cls(point, np.zeros([n, n]))

    @classmethod
    def uniform(cls) -> "MultivariateNormal":
        "A uniform distribution"
        return cls(np.zeros(0), np.zeros([0, 0]))
