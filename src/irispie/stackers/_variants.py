"""
# Stacked-time linear system

$$
\tilda x_t = 
$$

"""


#[

from __future__ import annotations

import numpy as _np

from ..fords.solutions import Solution

from ._invariants import Invariant

#]


mat_pow = _np.linalg.matrix_power


class Variant:
    """
    """

    __slots__ = (
        "TT", "PP", "RR", "KK",
        "AA", "BB", "CC", "DD", "HH",
    )

    def __init__(self, ) -> None:
        """
        """
        for n in self.__slots__:
            setattr(self, n, None, )

    @classmethod
    def from_solution(
        klass,
        invariant: Invariant,
        solution: Solution,
    ) -> Self:
        """
        """
        self = klass()
        num_periods = invariant.num_periods
        T, P, _, K, Z, H, D, = solution.unpack_square_solution()
        forward = num_periods - 1
        R = solution.expand_square_solution(forward=forward, )
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
        Z = Z[index_y, :]
        H = H[index_y, :][: , index_w]
        D = D[index_y]
        #
        TT = []
        KK = []
        PP = []
        RR = []
        #
        AA = []
        BB = []
        CC = []
        DD = []
        HH = []
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
            TT.append(full_TT[invariant.index_xi, :])
            KK.append(full_KK[invariant.index_xi])
            PP.append(full_PP[invariant.index_xi, :])
            RR.append(full_RR[invariant.index_xi, :])
            #
            AA.append(Z @ full_TT)
            BB.append(Z @ full_PP)
            CC.append(Z @ full_RR)
            DD.append(Z @ full_KK + D)
            HH.append(H)
            #
            Px = _move_forward(Px, num_u_included)
            Rx = _move_forward(Rx, num_v_included)
            Hx = _move_forward(Hx, num_w_included)
        #
        self.TT = _np.vstack(TT, )
        self.PP = _np.vstack(PP, )
        self.RR = _np.vstack(RR, )
        self.KK = _np.hstack(KK, )
        #
        self.AA = _np.vstack(AA, )
        self.BB = _np.vstack(BB, )
        self.CC = _np.vstack(CC, )
        self.DD = _np.hstack(DD, )
        self.HH = _np.vstack(HH, )
        #
        return self


def _move_forward(X: _np.ndarray, by: int, /, ) -> _np.ndarray:
    num_rows = X.shape[0]
    return _np.hstack([_np.zeros((num_rows, by)), X[:, :-by], ], )

