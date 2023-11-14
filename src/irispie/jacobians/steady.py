"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from typing import (Protocol, )
from collections.abc import (Iterable, )
import numpy as _np

from ..incidences import main as _incidence

from . import _base
#]


def _create_data_index_for_token(
    token: _incidence.Token,
    columns_to_eval: tuple[int, int],
    /,
) -> tuple[int, slice]:
    """
    """
    return (
        token.qid,
        slice(columns_to_eval[0], columns_to_eval[1]+1),
    )


def _create_eid_to_wrt_qids(
    equations: Iterable[_equations.Equation],
    all_wrt_qids: Iterable[int],
    /,
) -> dict[int, tuple[int, ...]]:
    """
    Create {eid: (qid, ... ), ...} where (qid, ... ) are qids wrt which the
    equation is defined no matter the time shift of the token
    """
    return {
        eqn.id: tuple(
            qid for qid in all_wrt_qids
            if _incidence.is_qid_in_tokens(eqn.incidence, qid, )
        )
        for eqn in equations
    }


class FlatSteadyJacobian(_base.Jacobian, ):
    """
    """
    #[

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, steady_array, )
        return self._create_jacobian(self._shape, diff_array, self._map, )

    _create_eid_to_wrt_something = \
        staticmethod(_create_eid_to_wrt_qids, )

    #
    # ===== Implement AtomFactoryProtocol =====
    # This protocol is used to manufacture aldi Atoms
    #

    create_data_index_for_token = \
        staticmethod(_create_data_index_for_token, )

    @staticmethod
    def create_diff_for_token(
        token: _incidence.Token,
        wrt_qids: tuple[int],
        /,
    ) -> _np.ndarray:
        """
        """
        try:
            index = wrt_qids.index(token.qid)
            diff = _np.zeros((len(wrt_qids), 1))
            diff[index] = 1
            return diff
        except:
            return 0

    #]


class NonflatSteadyJacobian(_base.Jacobian, ):
    """
    """
    #[

    NONFLAT_STEADY_SHIFT: int = 1

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, steady_array, )
        A = self._create_jacobian(self._shape, diff_array[:, 0:1], self._map, )
        B = self._create_jacobian(self._shape, diff_array[:, 1:2], self._map, )
        k = self.NONFLAT_STEADY_SHIFT
        return _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))

    _create_eid_to_wrt_something = \
        staticmethod(_create_eid_to_wrt_qids, )

    #
    # ===== Implement AtomFactoryProtocol =====
    # This protocol is used to manufacture aldi Atoms
    #

    create_data_index_for_token = \
        staticmethod(_create_data_index_for_token, )

    @staticmethod
    def create_diff_for_token(
        token: _incidence.Token,
        wrt_qids: dict[int, tuple[int]],
        /,
    ) -> _np.ndarray:
        """
        """
        try:
            index = wrt_qids.index(token.qid)
            diff = _np.zeros((len(wrt_qids), 2))
            diff[index, :] = 1, token.shift
            return diff
        except:
            return 0

    #]

