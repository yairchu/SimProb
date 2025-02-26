"""
Multivariate normal (Gaussian) distributions.

Unlike scipy.stats.multivariate_normal, MultivariateNormal implements many useful methods.
"""

import dataclasses
import numpy as np
import scipy


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
        if self.dim < other.shape[1]:
            return other @ self.extend(other.shape[1])
        return type(self)(mean=other @ self.mean, covar=other @ self.covar @ other.T)

    def __and__(self, other: "MultivariateNormal") -> "MultivariateNormal":
        "Fuse two Gaussian distributions modelling the same random variable."
        if other.dim < self.dim:
            other = other.extend(self.dim)
        elif self.dim < other.dim:
            return other & self

        diff = other - self  # (y)
        kalman_gain = PartialCovar.solve(diff.covar, self.covar).T  # (K)
        return type(self)(
            mean=self.mean + kalman_gain.real @ diff.mean,
            covar=(np.eye(self.dim) - kalman_gain) @ self.covar,
        )

    def extend(self, dim: int) -> "MultivariateNormal":
        add = dim - self.dim
        return type(self)(
            np.pad(self.mean, (0, add)), PartialCovar.extend(self.covar, dim)
        )

    @classmethod
    def delta(cls, n: int) -> "MultivariateNormal":
        return cls(np.zeros(n), np.zeros([n, n]))


def assure_symmetric_matrix(mat, do_assert=False):
    "Enforce matrix symmetry (for covariance matrices)"
    res = 0.5 * (mat + mat.T)
    if do_assert:
        assert np.isclose(mat, res, rtol=0.01).all(), (
            f"Matrix is not symmetric.\n{mat=}"
        )
    return res


@dataclasses.dataclass
class PartialCovar:
    """
    Covariance matrix extended to support uniform distributions variance.

    The extension represents the components of the covariance which are infinite.
    In other terms, it spans the sub-space that which has a uniform distribution.
    """

    real: np.ndarray
    ext: np.ndarray

    def __post_init__(self):
        assert self.shape == (self.dim, self.dim)
        assert self.ext.shape == self.shape

    @property
    def shape(self):
        return self.real.shape

    @property
    def dim(self):
        return self.shape[0]

    def emulate_as_real(self, inf_const=1e6):
        """
        Create a real matrix "emulation" of the partial covariance
        by multiplying the extended part by a large number.

        This allows performing sanity tests,
        like verifying the solve method gives close results to the emulated ones.
        """
        return inf_const * self.ext + self.real

    def simplify(self):
        return self.real if np.allclose(self.ext, 0) else self

    @classmethod
    def extend(cls, covar, dim):
        "Add uniform distribution dimensions to a given covariance matrix"
        prev = covar.real.shape[0]
        add = dim - prev
        if isinstance(covar, np.ndarray):
            covar = cls(covar, np.zeros_like(covar))
        pad = [(0, add)] * 2
        return cls(
            np.pad(covar.real, pad),
            np.pad(covar.ext, pad) + np.diag(np.pad(np.ones(add), (prev, 0))),
        )

    @classmethod
    def solve(cls, a, b, debug=True):
        "Compute X solving `a @ X = b`"
        if isinstance(a, np.ndarray) or isinstance(a.simplify(), np.ndarray):
            return np.linalg.solve(a.real, b)

        # Results should be very close to
        # return np.linalg.solve(a.emulate_as_real(), b if isinstance(b, np.ndarray) else b.emulate_as_real())

        null_a_ext = scipy.linalg.null_space(a.ext)
        span_a_ext = scipy.linalg.null_space(null_a_ext.T)

        def debug(emulate, our_res):
            print("A:\n", a)
            print("B:\n", b)
            print("Emulation:\n", emulate)
            print("Our calc:\n", our_res, "\n")

        if isinstance(b, np.ndarray):
            # Special case when many parts become zero.
            emulate = np.linalg.solve(a.emulate_as_real(), b)

            if debug:
                print("Special case (b is real)")
                # This appears to work incorrectly, fallback for now instead
                x_real_n, *_ = np.linalg.lstsq(a.real @ null_a_ext, b.real)
                debug(emulate, null_a_ext @ x_real_n)

            return emulate

        x_real_s = np.linalg.solve(
            span_a_ext.T @ np.linalg.solve(a.real, a.ext @ span_a_ext),
            span_a_ext.T @ np.linalg.solve(a.real, b.ext),
        )
        x_ext = (
            null_a_ext
            @ null_a_ext.T
            @ np.linalg.solve(a.real, b.ext - a.ext @ span_a_ext @ x_real_s)
        )

        null_b_ext = scipy.linalg.null_space(b.ext)
        span_b_ext = scipy.linalg.null_space(null_b_ext.T)
        x_real_n, *_ = np.linalg.lstsq(
            a.real @ null_a_ext,
            (b.real - a.real @ span_a_ext @ x_real_s) @ span_b_ext @ span_b_ext.T,
        )
        res = cls(span_a_ext @ x_real_s + null_a_ext @ x_real_n, x_ext).simplify()

        if null_b_ext.shape[1] > 0 and null_a_ext.shape[1] > 0:
            # Both A and B are rank-deficient.
            # In these cases our solution is incomplete according to simulation.
            # Fall back to emulation until that is fixed!
            emulate = np.linalg.solve(a.emulate_as_real(), b.emulate_as_real())

            if debug:
                print("Falling back")
                debug(emulate, res)

            return emulate

        if debug:
            print("Proper PartialCover solve!")
        return res

    def __add__(self, other):
        return type(self)(
            self.real + other.real,
            (self.ext + other.ext) if isinstance(other, PartialCovar) else self.ext,
        ).simplify()

    __array_priority__ = 1000  # Ensures NumPy prefers calling our roperators

    def __radd__(self, other):
        return self + other

    def __matmul__(self, other: np.ndarray) -> "PartialCovar":
        return type(self)(self.real @ other, self.ext @ other).simplify()

    def __rmatmul__(self, other: np.ndarray) -> "PartialCovar":
        return type(self)(other @ self.real, other @ self.ext).simplify()

    # needed for assure_symmetric_matrix
    @property
    def T(self):
        return type(self)(self.real.T, self.ext.T)

    # needed for assure_symmetric_matrix
    def __rmul__(self, scalar):
        return type(self)(scalar * self.real, scalar * self.ext).simplify()
