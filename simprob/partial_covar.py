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
    * [S, N] bounded_subspace: The sub-space that has a bounded distribution
    * [S, S] sub_covar: The covariance of the bounded subspace
    """

    bounded_subspace: np.ndarray = dataclasses.field(kw_only=True)
    sub_covar: np.ndarray = dataclasses.field(kw_only=True)

    def __post_init__(self):
        self.sub_covar = self.assure_symmetric_matrix(self.sub_covar)

    @property
    def dim(self):
        return self.bounded_subspace.shape[1]

    @property
    def shape(self):
        return (self.dim,) * 2

    @property
    def uniform_subspace(self):
        return scipy.linalg.null_space(self.bounded_subspace).T

    @property
    def real(self):
        """
        Real component of the covariance projected to the full covariance matrix
        (uniform parts are zeroed).
        """
        return PartialCovar.transform(self.sub_covar, self.bounded_subspace.T)

    def emulate_as_real(self, inf_const=1e6):
        """
        Create a real matrix "emulation" of the partial covariance
        by representing uniform variance as a very large variance.
        """
        return self.real + inf_const * self.uniform_subspace.T @ self.uniform_subspace

    def simplify(self):
        return self.real if self.bounded_subspace.shape == self.shape else self

    @classmethod
    def uniform(cls, dim):
        return cls(
            bounded_subspace=np.zeros([0, dim]), sub_covar=np.zeros([0, 0])
        ).simplify()

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
            bounded_subspace=np.concatenate(projected_spaces),
            sub_covar=scipy.linalg.block_diag(
                *[x if isinstance(x, np.ndarray) else x.sub_covar for x in covars]
            ),
        ).simplify()

    @classmethod
    def inv(cls, mat):
        bounded_subspace = (
            np.eye(mat.shape[0])
            if isinstance(mat, np.ndarray)
            else mat.bounded_subspace
        )
        sub = mat if isinstance(mat, np.ndarray) else mat.sub_covar
        delta_space = scipy.linalg.null_space(sub).T
        if delta_space.shape[0] == 0:
            inv_part = np.linalg.inv(sub)
        else:
            var_space = scipy.linalg.null_space(delta_space).T
            inv_part = cls(
                bounded_subspace=var_space,
                sub_covar=np.linalg.inv(cls.transform(sub, var_space)),
            )
        return cls.transform(inv_part, bounded_subspace.T)

    @staticmethod
    def transform(covar, mat):
        real = mat @ covar.real @ mat.T
        if isinstance(covar, np.ndarray):
            return real
        subspace = scipy.linalg.null_space(covar.uniform_subspace @ mat.T).T
        return type(covar)(
            bounded_subspace=subspace,
            sub_covar=PartialCovar.transform(real, subspace),
        )

    def __add__(self, other):
        uni_spaces = [
            x.uniform_subspace for x in [self, other] if not isinstance(x, np.ndarray)
        ]
        subspace = scipy.linalg.null_space(np.concatenate(uni_spaces)).T
        return type(self)(
            bounded_subspace=subspace,
            sub_covar=PartialCovar.transform(self.real + other.real, subspace),
        ).simplify()

    __array_priority__ = 1000  # Ensures NumPy prefers calling our roperators

    def __radd__(self, other):
        return self + other

    @staticmethod
    def assure_symmetric_matrix(mat, do_assert=False):
        "Enforce matrix symmetry (for covariance matrices)"
        if isinstance(mat, PartialCovar):
            return dataclasses.replace(
                mat,
                sub_covar=PartialCovar.assure_symmetric_matrix(
                    mat.sub_covar, do_assert
                ),
            )
        res = 0.5 * (mat + mat.T)
        if do_assert:
            assert np.isclose(mat, res, rtol=0.01).all(), (
                f"Matrix is not symmetric.\n{mat=}"
            )
        return res
