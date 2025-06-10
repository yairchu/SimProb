"""
Multivariate normal (Gaussian) distributions.

Unlike scipy.stats.multivariate_normal, MultivariateNormal implements many useful methods.

Also supports uniform distributions by using a SubspaceMetric as its covariance matrix.

Properties: (TODO: test with hypothesis)
* fuse/`&` is associative and cummutative, with uniform as its identity element
* `+` is associative and cummutative, with delta(zeros(dim)) as its identity element
* Distributive properties:
  * (m @ a) & (m @ b) = m @ (a & b)
  * (m @ a) + (m @ b) = m @ (a + b)
"""

import numpy as np

from .subspace_metric import SubspaceMetric


class MultivariateNormal:
    "A multivariate normal (Gaussian) distribution."

    def __init__(
        self,
        *,
        mean=None,
        covar=None,
        info_vector=None,
        precision=None,
    ):
        assert (mean is None) != (info_vector is None), (
            "Either mean or info_vector must be provided, but not both."
        )
        assert (covar is None) != (precision is None), (
            "Either covar or precision must be provided, but not both."
        )
        self._mean = mean
        self._covar = covar
        self._info_vector = info_vector
        self._precision = precision

        m = precision if covar is None else covar
        assert m.shape == (self.dim, self.dim), (
            "Covariance/precision matrix shape mismatch."
        )

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.linalg.solve(self.precision, self.info_vector)
        return self._mean

    @property
    def info_vector(self):
        if self._info_vector is None:
            self._info_vector = self.precision @ self.mean
        return self._info_vector

    @property
    def precision(self):
        if self._precision is None:
            self._precision = SubspaceMetric.inv(self.covar)
        return self._precision

    @property
    def covar(self):
        if self._covar is None:
            self._covar = SubspaceMetric.inv(self.precision)
        return self._covar

    @property
    def dim(self) -> int:
        "The number of dimensions of the distribution."
        return (self._info_vector if self._mean is None else self._mean).shape[0]

    def __add__(self, other) -> "MultivariateNormal":
        """
        Compute the distribution of the sum of two independent Gaussian-distributed variables.

        Summation forms a cummutative group (https://en.wikipedia.org/wiki/Abelian_group).
        """
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
            mean=other @ self.mean, covar=SubspaceMetric.transform(self.covar, other)
        )

    def __and__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Fuse two distributions modeling the same random variable."
        return type(self).fuse([self, other])

    @classmethod
    def fuse(cls, dists):
        """
        Fuse distributions modeling the same random variable.

        Fusion forms a cummutative group (https://en.wikipedia.org/wiki/Abelian_group).
        """
        dists = cls.broadcast_dists(dists)
        return cls(
            info_vector=sum(x.info_vector for x in dists),
            precision=sum(x.precision for x in dists),
        )

    @classmethod
    def delta(cls, point: np.ndarray) -> "MultivariateNormal":
        """
        A point distribution with no variance (the value is known exactly).

        delta of zeros is the identity element for summation:

            delta(zeros(dim)) + x == x
        """
        [n] = point.shape
        return cls(mean=point, covar=np.zeros([n, n]))

    @classmethod
    def uniform(cls, dim=0) -> "MultivariateNormal":
        """
        A uniform distribution (nothing in known).

        The uniform distribution is the identity element for fusion:

            uniform() & x == x
        """
        return cls(mean=np.zeros(dim), precision=np.zeros([dim, dim]))

    @classmethod
    def broadcast_dists(cls, dists):
        "Broadcasts distributions to same dimensionality"
        # Convert state vectors into point-distributions
        dists = [cls.delta(x) if isinstance(x, np.ndarray) else x for x in dists]
        max_dim = max(x.dim for x in dists)
        return [x.extend(max_dim) for x in dists]

    def extend(self, dim: int) -> "MultivariateNormal":
        "Extend distribution to given dimension (adding unknown/uniformly-distributed dimensions)"
        return type(self).concat([self, self.uniform(dim - self.dim)])

    @classmethod
    def concat(cls, dists):
        "Concatenate distributions over distinct variables to a joint distribution over their concatenation"
        return cls(
            mean=np.concatenate([x.mean for x in dists]),
            covar=SubspaceMetric.concat([x.covar for x in dists]),
        )
