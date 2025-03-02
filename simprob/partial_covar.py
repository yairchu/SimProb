"""
Covariance matrices extended to support uniform distributions.

In Kalman filters where observations measure only part of the state,
a PartialCovar is used to signify that the other parts are not known in the observation.
"""

import dataclasses
import numpy as np
import scipy


@dataclasses.dataclass
class PartialCovar:
    """
    Covariance matrix extended to support uniform distributions variance.

    Components:
    * real: The real component of the full covariance matrix (removing infinite terms)
    * bounded_subspace: The sub-space that has a bounded distribution
    """

    real: np.ndarray
    bounded_subspace: np.ndarray = dataclasses.field(kw_only=True)

    def __post_init__(self):
        self.real = self.assure_symmetric_matrix(self.real)
        assert self.bounded_subspace.shape[1] == self.dim, (
            f"{self.bounded_subspace.shape=} {self.dim=}"
        )

    @property
    def shape(self):
        return self.real.shape

    @property
    def dim(self):
        return self.shape[0]

    @property
    def uniform_subspace(self):
        return scipy.linalg.null_space(self.bounded_subspace).T

    def emulate_as_real(self, inf_const=1e6):
        """
        Create a real matrix "emulation" of the partial covariance
        by multiplying the extended part by a large number.

        This allows performing sanity tests,
        like verifying the solve method gives close results to the emulated ones.
        """
        return inf_const * self.uniform_subspace.T @ self.uniform_subspace + self.real

    def simplify(self):
        return self.real if self.bounded_subspace.shape == self.shape else self

    @classmethod
    def extend(cls, covar, dim):
        "Add uniform distribution dimensions to a given covariance matrix"
        prev = covar.real.shape[0]
        if isinstance(covar, np.ndarray):
            covar = cls(covar, bounded_subspace=np.eye(prev))
        add = dim - prev
        if add == 0:
            return covar
        return cls(
            np.pad(covar.real, [(0, add)] * 2),
            bounded_subspace=np.pad(covar.bounded_subspace, [(0, 0), (0, add)]),
        )

    @classmethod
    def inv(cls, mat):
        if isinstance(mat, np.ndarray):
            mat = cls.extend(mat, len(mat))
        bounded_part = cls.transform(mat.real, mat.bounded_subspace)
        delta_space = scipy.linalg.null_space(bounded_part)
        if delta_space.shape[1] == 0:
            inv_part = np.linalg.inv(bounded_part)
        else:
            var_space = scipy.linalg.null_space(delta_space.T)
            inv_part = cls(
                cls.transform(
                    np.linalg.inv(cls.transform(bounded_part, var_space.T)), var_space
                ),
                bounded_subspace=var_space.T,
            )
        return cls.transform(inv_part, mat.bounded_subspace.T)

    @staticmethod
    def transform(covar, mat):
        real = mat @ covar.real @ mat.T
        if isinstance(covar, np.ndarray):
            return real

        return type(covar)(
            real,
            bounded_subspace=scipy.linalg.null_space(covar.uniform_subspace @ mat.T).T,
        )

    def __add__(self, other):
        if not isinstance(other, PartialCovar) or other.dim < self.dim:
            other = type(self).extend(other, self.dim)
        if self.dim < other.dim:
            return other + self
        return type(self)(
            self.real + other.real,
            bounded_subspace=scipy.linalg.null_space(
                np.concatenate([self.uniform_subspace, other.uniform_subspace])
            ).T,
        ).simplify()

    __array_priority__ = 1000  # Ensures NumPy prefers calling our roperators

    def __radd__(self, other):
        return self + other

    @staticmethod
    def assure_symmetric_matrix(mat, do_assert=False):
        "Enforce matrix symmetry (for covariance matrices)"
        if isinstance(mat, PartialCovar):
            return dataclasses.replace(
                mat, real=PartialCovar.assure_symmetric_matrix(mat.real, do_assert)
            )
        res = 0.5 * (mat + mat.T)
        if do_assert:
            assert np.isclose(mat, res, rtol=0.01).all(), (
                f"Matrix is not symmetric.\n{mat=}"
            )
        return res
