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
        [a, b] = type(self).broadcast_dists([self, other])
        return type(self)(mean=a.mean + b.mean, covar=a.covar + b.covar)

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
        return type(self).fuse([self, other])

    @classmethod
    def fuse(cls, dists):
        "Fuse two Gaussian distributions modeling the same random variable."
        invs_and_means = [
            (PartialCovar.inv(x.covar), x.mean) for x in cls.broadcast_dists(dists)
        ]
        covar = PartialCovar.inv(sum(inv for inv, _ in invs_and_means))
        return cls(covar @ sum(inv @ mean for inv, mean in invs_and_means), covar)

    def extend(self, dim: int) -> "MultivariateNormal":
        return type(self).concat([self, self.uniform(dim - self.dim)])

    @classmethod
    def concat(cls, dists):
        "Concatenate distributions over distinct variables to a joint distribution over their concatenation"
        return cls(
            np.concatenate([x.mean for x in dists]),
            PartialCovar.concat([x.covar for x in dists]),
        )

    @classmethod
    def delta(cls, point: np.ndarray) -> "MultivariateNormal":
        "A point distribution (value is completely known)"
        [n] = point.shape
        return cls(point, np.zeros([n, n]))

    @classmethod
    def uniform(cls, dim=0) -> "MultivariateNormal":
        "A uniform distribution"
        return cls(np.zeros(dim), PartialCovar.uniform(dim))

    @classmethod
    def broadcast_dists(cls, dists):
        "Broadcasts distributions to same dimensionality"
        # Convert state vectors into point-distributions
        dists = [cls.delta(x) if isinstance(x, np.ndarray) else x for x in dists]
        max_dim = max(x.dim for x in dists)
        return [x.extend(max_dim) for x in dists]
