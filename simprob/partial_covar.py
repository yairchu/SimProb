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
        by representing uniform variance as a very large variance.
        """
        return inf_const * self.uniform_subspace.T @ self.uniform_subspace + self.real

    def simplify(self):
        return self.real if self.bounded_subspace.shape == self.shape else self

    @classmethod
    def uniform(cls, dim):
        return cls(np.zeros([dim, dim]), bounded_subspace=np.zeros([0, dim])).simplify()

    @classmethod
    def concat(cls, covars):
        """
        Covariance of joint distribution of distinct variables.

        Generalization of scipy.linalg.block_diag to partial covariances
        """
        spaces = [
            np.eye(x.shape[0]) if isinstance(x, np.ndarray) else x.bounded_subspace
            for x in covars
        ]
        cum_dims = np.cumsum([x.shape[0] for x in covars])
        total_dims = cum_dims[-1]
        projected_spaces = [
            np.pad(s, [(0, 0), (dims - s.shape[1], total_dims - dims)])
            for dims, s in zip(cum_dims, spaces)
        ]
        return cls(
            scipy.linalg.block_diag(*[x.real for x in covars]),
            bounded_subspace=np.concatenate(projected_spaces),
        ).simplify()

    @classmethod
    def inv(cls, mat):
        bounded_subspace = (
            np.eye(mat.shape[0])
            if isinstance(mat, np.ndarray)
            else mat.bounded_subspace
        )
        bounded_part = cls.transform(mat.real, bounded_subspace)
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
        return cls.transform(inv_part, bounded_subspace.T)

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
        uni_spaces = [
            x.uniform_subspace for x in [self, other] if not isinstance(x, np.ndarray)
        ]
        return type(self)(
            self.real + other.real,
            bounded_subspace=scipy.linalg.null_space(np.concatenate(uni_spaces)).T,
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
