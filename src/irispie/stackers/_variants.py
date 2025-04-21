"""
# Stacked-time linear system

$$
\tilda x_t = 
$$

"""


#[

from __future__ import annotations

import numpy as _np

from ..fords import initializers as _initializers
from ..fords.solutions import Solution

from ._invariants import Invariant

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from numbers import Real

#]


mat_pow = _np.linalg.matrix_power


class Variant:
    """
    """

    _system_matrix_slots = (
        "TT", "PP", "RR",
        "AA", "BB", "CC", "HH",
    )

    _system_intercept_slots = (
        "KK", "DD",
    )

    __slots__ = (
        *_system_matrix_slots,
        *_system_intercept_slots,
        "init_med",
        "init_mse",
        "Xi",
        "MM",
        "std_name_to_value",
        "U",
        "has_unknown_initial",
    )

    def __init__(self, ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )

    @classmethod
    def from_solution_and_stds(
        klass,
        invariant: Invariant,
        solution: Solution,
        std_name_to_value: dict[str, Real],
        cov_u: _np.ndarray,
    ) -> Self:
        """
        """
        self = klass()
        #
        num_periods = invariant.num_periods
        forward = num_periods - 1
        T, P, _, K, Z, H, D, U = solution.unpack_triangular_solution()
        R = solution.expand_triangular_solution(forward=forward, )
        index_xi = invariant.index_xi
        index_u = invariant.index_u
        index_v = invariant.index_v
        index_y = invariant.index_y
        index_w = invariant.index_w
        num_xi = T.shape[0]
        num_u = P.shape[1]
        num_v = R[0].shape[1]
        num_y_included = len(index_y)
        num_u_included = len(index_u)
        num_v_included = len(index_v)
        num_w_included = len(index_w)
        Px = _np.hstack([P[:, index_u], _np.zeros((num_xi, num_u_included*(num_periods-1)))], )
        Rx = _np.hstack([r[:, index_v] for r in R])
        Hx = _np.hstack([H[index_y, :][:, index_w], _np.zeros((num_y_included, num_w_included*(num_periods-1)))], )
        U = U[index_xi, :]
        Z = Z[index_y, :]
        D = D[index_y]
        #
        for n in self._system_matrix_slots:
            setattr(self, n, [], )
        for n in self._system_intercept_slots:
            setattr(self, n, [], )
        #
        full_TT = _np.eye(num_xi, )
        full_KK = _np.zeros((num_xi, ), )
        full_PP = _np.zeros((num_xi, num_u_included*num_periods, ), )
        full_RR = _np.zeros((num_xi, num_v_included*num_periods, ), )
        for k in range(num_periods):
            full_TT = T @ full_TT
            full_KK = T @ full_KK + K
            full_PP = T @ full_PP + Px
            full_RR = T @ full_RR + Rx
            #
            self.TT.append(U @ full_TT)
            self.KK.append(U @ full_KK)
            self.PP.append(U @ full_PP)
            self.RR.append(U @ full_RR)
            #
            self.AA.append(Z @ full_TT)
            self.BB.append(Z @ full_PP)
            self.CC.append(Z @ full_RR)
            self.DD.append(Z @ full_KK + D)
            self.HH.append(Hx)
            #
            Px = _move_forward(Px, num_u_included)
            Rx = _move_forward(Rx, num_v_included)
            Hx = _move_forward(Hx, num_w_included)
        #
        for n in self._system_matrix_slots:
            a = getattr(self, n)
            setattr(self, n, _np.vstack(a, ), )
        for n in self._system_intercept_slots:
            a = getattr(self, n)
            setattr(self, n, _np.hstack(a, ), )
        self.U = U
        #
        self.std_name_to_value = std_name_to_value
        #
        self.init_med, self.init_mse, Xi = \
            _initializers.initialize(solution, cov_u, )
        #
        self.has_unknown_initial = Xi is not None
        if self.has_unknown_initial:
            self.Xi = Xi
            self.MM = self.AA @ Xi
        #
        return self


def _move_forward(X: _np.ndarray, by: int, /, ) -> _np.ndarray:
    num_rows = X.shape[0]
    return _np.hstack([_np.zeros((num_rows, by)), X[:, :-by], ], )

