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


class PeriodJacobian(_base.Jacobian, ):
    """
    """
    #[

    def eval(
        self,
        data_array: _np.ndarray,
        steady_array: _np.ndarray | None,
        /,
        column_offset: int | None = None,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(
            data_array, steady_array,
            column_offset=column_offset,
        )
        return self._create_jacobian(self._shape, diff_array, self._map, )

    @staticmethod
    def _create_eid_to_wrt_something(
        equations: Iterable[_equations.Equation],
        all_wrt_qids: Iterable[int],
        /,
    ) -> dict[int, tuple[int, ...]]:
        """
        """
        return {
            eqn.id: tuple(
                qid for qid in all_wrt_qids
                if _incidence.is_qid_zero_in_tokens(eqn.incidence, qid, )
            )
            for eqn in equations
        }

    #
    # ===== Implement AtomFactoryProtocol =====
    # This protocol is used to manufacture aldi Atoms
    #

    @staticmethod
    def create_data_index_for_token(
        token: _incidence.Token,
        columns_to_eval: tuple[int, int],
        /,
    ) -> tuple[int, slice]:
        """
        """
        # column_index = _np.arange(
            # columns_to_eval[0]+token.shift,
            # columns_to_eval[1]+token.shift+1,
        # )
        column_index = columns_to_eval[0]+token.shift
        return (token.qid, column_index, )

    @staticmethod
    def create_diff_for_token(
        token: _incidence.Token,
        wrt_qids: tuple[int],
        /,
    ) -> _np.ndarray:
        """
        """
        try:
            if token.shift:
                return 0
            index = wrt_qids.index(token.qid)
            diff = _np.zeros((len(wrt_qids), 1))
            diff[index] = 1
            return diff
        except:
            return 0

    #]

