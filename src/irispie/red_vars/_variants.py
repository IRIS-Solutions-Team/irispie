"""
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp
from numbers import Number

from ..dataslates import Dataslate
from ..fords.solutions import Solution
from ._invariants import Invariant

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Iterable
    from ..dates import Period

#]


class System:
    r"""
    """
    #[

    __slots__ = (
        "A",
        "B",
        "c",
        "cov_residuals",
    )

    def __init__(
        self,
        A: _np.ndarray | None = None,
        B: _np.ndarray | None = None,
        c: _np.ndarray | None = None,
        cov_residuals: _np.ndarray | None = None,
    ) -> None:
        r"""
        """
        self.A = A
        self.B = B
        self.c = c
        self.cov_residuals = cov_residuals

    def copy(self, ) -> Self:
        r"""
        """
        new = type(self)()
        for n in self.__slots__:
            attr = getattr(self, n)
            if attr is not None:
                setattr(new, n, attr.copy())
        return new

    @property
    def _can_calculate_dimensions(self, ) -> bool:
        r"""
        """
        return self.A is not None

    @property
    def num_endogenous(self, ) -> int:
        r"""
        """
        if self._can_calculate_dimensions:
            return self.A.shape[0]
        else:
            return None

    @property
    def order(self, ) -> int:
        r"""
        """
        if self._can_calculate_dimensions:
            return self.A.shape[1] // self.num_endogenous
        else:
            return None

    @property
    def has_intercept(self, ) -> bool:
        r"""
        """
        return self.c is not None

    @property
    def num_exogenous(self, ) -> int:
        r"""
        """
        if self._can_calculate_dimensions:
            return self.B.shape[1]
        else:
            return None

    @property
    def num_lagged_endogenous(self, ) -> int:
        r"""
        """
        if self._can_calculate_dimensions:
            return self.num_endogenous * self.order
        else:
            return None

    #]


class Variant:
    """
    """
    #[

    __slots__ = (
        "system",
        "fitted_periods",
        "residual_estimates",

        "_companion_T",
        "_eigenvalues",
        "_max_abs_eigenvalue",
    )

    def __init__(
        self,
        A: _np.ndarray | None = None,
        B: _np.ndarray | None = None,
        c: _np.ndarray | None = None,
        cov_residuals: _np.ndarray | None = None,
        fitted_periods: Iterable[Period] | None = (),
        residual_estimates: _np.ndarray | None = None,
        where_observations: _np.ndarray | None = None,
    ) -> None:
        r"""
        """
        self.system = System(A=A, B=B, c=c, cov_residuals=cov_residuals, )
        self.fitted_periods = tuple(fitted_periods)
        self.residual_estimates = residual_estimates
        #
        self._eigenvalues = None
        self._max_abs_eigenvalue = None
        self._companion_T = None

    def copy(self, ) -> Self:
        r"""
        """
        new = type(self)()
        new.system = self.system.copy()
        new.fitted_periods = self.fitted_periods
        new._eigenvalues = self._eigenvalues
        new._max_abs_eigenvalue = self._max_abs_eigenvalue
        new._companion_T = self._companion_T
        return new

    @property
    def eigenvalues(self, ) -> tuple[Number, ...]:
        r"""
        """
        if self._eigenvalues is None:
            self._populate_eigenvalues()
        return self._eigenvalues

    def _populate_eigenvalues(self, ) -> None:
        r"""
        """
        T = self.companion_T
        if T is None:
            self._eigenvalues = None
            self._max_abs_eigenvalue = None
            return
        eigenvalues = _np.linalg.eigvals(T)
        max_abs_eigenvalue = _np.max(_np.abs(eigenvalues))
        self._eigenvalues = _tuple_from_flat_array(eigenvalues, )
        self._max_abs_eigenvalue = _number_from_numpy(max_abs_eigenvalue, )

    @property
    def max_abs_eigenvalue(self, ) -> Number:
        r"""
        """
        if self._max_abs_eigenvalue is None:
            self._populate_eigenvalues()
        return self._max_abs_eigenvalue

    @property
    def is_stable(self, ) -> bool:
        r"""
        """
        return (
            self.max_abs_eigenvalue < 1
            if self.max_abs_eigenvalue is not None
            else None
        )

    @property
    def companion_T(self, ) -> _np.ndarray:
        r"""
        """
        if self._companion_T is None:
            self._populate_companion_T()
        return self._companion_T

    def _populate_companion_T(self, ) -> None:
        r"""
        """
        if self.system.A is None:
            self._companion_T = None
            return
        num_endogenous = self.system.num_endogenous
        order = self.system.order
        num_extra = num_endogenous * (order - 1)
        dynamic_identity = _np.eye(num_extra, num_endogenous * order, )
        self._companion_T = _np.concatenate((
            self.system.A,
            dynamic_identity,
        ))

    def _get_companion_P(self, ) -> _np.ndarray:
        r"""
        """
        num_endogenous = self.system.num_endogenous
        num_lagged_endogenous = self.system.num_lagged_endogenous
        return _np.eye(num_lagged_endogenous, num_endogenous, dtype=float, )

    def _get_companion_K(self, ) -> _np.ndarray:
        r"""
        """
        num_endogenous = self.system.num_endogenous
        num_lagged_endogenous = self.system.num_lagged_endogenous
        if self.system.c is None:
            return _np.zeros((num_lagged_endogenous, ), dtype=float, )
        else:
            return _np.concatenate((
                self.system.c,
                _np.zeros((num_lagged_endogenous - num_endogenous, ), dtype=float, ),
            ))

    def _get_companion_sigma(self, ) -> tuple[_np.ndarray, _np.ndarray, ]:
        r"""
        """
        num_endogenous = self.system.num_endogenous
        order = self.system.order
        num_extra = num_endogenous * (order - 1)
        extra_zeros = _np.zeros((num_extra, num_extra), dtype=self.system.cov_residuals.dtype, )
        return _sp.linalg.block_diag(
            self.system.cov_residuals, extra_zeros,
        )

    def get_acov(
        self,
        up_to_order: int = 0,
    ) -> _np.ndarray:
        r"""
        """
        num_endogenous = self.system.num_endogenous
        companion_T = self.companion_T
        companion_sigma = self._get_companion_sigma()
        Omega = _sp.linalg.solve_discrete_lyapunov(companion_T, companion_sigma, )
        acov = []
        acov.append(Omega[:num_endogenous, :num_endogenous])
        for i in range(up_to_order):
            Omega = companion_T @ Omega
            acov.append(Omega[:num_endogenous, :num_endogenous])
        return tuple(acov)

    def get_mean(self, ) -> _np.ndarray:
        r"""
        """
        num_endogenous = self.system.num_endogenous
        order = self.system.order
        A = self.system.A
        c = self.system.c
        if c is None or _np.all(c == 0):
            return _np.zeros((num_endogenous, ), dtype=A.dtype, )
        new_shape = (num_endogenous, num_endogenous, order, )
        A = A.reshape(new_shape, order="F", )
        A = A.sum(axis=2, )
        I = _np.eye(num_endogenous, )
        return _np.linalg.solve(I - A, c)

    def _get_companion_solution(
        self,
        deviation: bool = False,
    ) -> Solution:
        r"""
        """
        solution = Solution(
            T=self.companion_T,
            P=self._get_companion_P(),
            K=self._get_companion_K(),
        )
        if deviation:
            solution = Solution.deviation_solution(solution, )
        return solution

    #]


def _tuple_from_flat_array(array: _np.ndarray, ) -> tuple[Number, ...]:
    r"""
    """
    return tuple(
        _number_from_numpy(i)
        for i in array.flatten()
    )


def _number_from_numpy(x) -> Number:
    r"""
    """
    return float(_np.real(x)) if _np.isreal(x) else complex(x)

