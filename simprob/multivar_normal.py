"""
Multivariate normal (Gaussian) distributions.

Unlike scipy.stats.multivariate_normal, MultivariateNormal implements many useful methods.
"""

import dataclasses
import numpy as np


def assure_symmetric_matrix(mat, do_assert=False):
    "Enforce matrix symmetry (for covariance matrices)"
    res = 0.5 * (mat + mat.T)
    if do_assert:
        assert np.isclose(mat, res, rtol=0.01).all(), (
            f"Matrix is not symmetric.\n{mat=}"
        )
    return res


@dataclasses.dataclass
class MultivariateNormal:
    "A multivariate normal (Gaussian) distribution."

    mean: np.ndarray  # Mean vector
    covar: np.ndarray  # Covariance matrix

    def __post_init__(self):
        assert self.covar.shape == (self.dim, self.dim), (
            "Covariance matrix shape mismatch."
        )
        self.covar = assure_symmetric_matrix(self.covar)

    @property
    def dim(self) -> int:
        "The number of dimensions of the distribution."
        return self.mean.shape[0]

    def __add__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Compute the distribution of the sum of two independent Gaussian-distributed variables."
        return type(self)(mean=self.mean + other.mean, covar=self.covar + other.covar)

    def __sub__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Compute  the distribution of the difference of two independent Gaussian-distributed variables."
        return self + (-other)

    def __neg__(self) -> "MultivariateNormal":
        "The distribution of the negation of a Gaussian-distributed variable."
        return type(self)(mean=-self.mean, covar=self.covar)

    __array_priority__ = 1000  # Ensures NumPy prefers calling our `__rmatmul__`

    def __rmatmul__(self, other: np.ndarray) -> "MultivariateNormal":
        "Apply a linear transformation to the distribution."
        return type(self)(mean=other @ self.mean, covar=other @ self.covar @ other.T)

    def __and__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Fuse two Gaussian distributions modelling the same random variable."
        return MultivariateNormalSubspace(self, np.eye(self.dim)) & other


@dataclasses.dataclass
class MultivariateNormalSubspace:
    dist: MultivariateNormal  # (z)
    subspace: np.ndarray  # (H)

    def __and__(self, other: MultivariateNormal) -> MultivariateNormal:
        "Fuse a distribution with a distribution of a subspace."

        assert self.subspace.shape == (self.dist.dim, other.dim)
        diff = self.dist - self.subspace @ other  # (y)
        # This is the only part in this module which is formula heavy and I personally do not find intuitive
        kalman_gain = np.linalg.solve(diff.covar, self.subspace @ other.covar).T  # (K)
        assert kalman_gain.shape == (other.dim, self.dist.dim)
        return type(other)(
            mean=other.mean + kalman_gain @ diff.mean,
            covar=(np.eye(other.dim) - kalman_gain @ self.subspace) @ other.covar,
        )
