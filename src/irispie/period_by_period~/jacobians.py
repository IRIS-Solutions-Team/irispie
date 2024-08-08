"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from typing import (Protocol, )
from collections.abc import (Iterable, )
import numpy as _np

from ..incidences import main as _incidence

from ..jacobians import base
#]


class PeriodJacobian(base.Jacobian, ):
    """
    """
    #[

    def eval(
        self,
        data_array: _np.ndarray,
        column_offset: int,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(data_array, column_offset, )
        return self._create_jacobian_matrix(diff_array, )

    def _create_eid_to_wrts(
        self,
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

    # ===== Implement AtomFactoryProtocol =====

    def create_data_index_for_token(
        self,
        token: _incidence.Token,
        /,
    ) -> tuple[int, slice]:
        """
        """
        return (token.qid, token.shift, )

    def create_diff_for_token(
        self,
        token: _incidence.Token,
        wrt_qids: tuple[int],
        /,
    ) -> _np.ndarray:
        """
        """
        if token is None:
            return _np.zeros((len(wrt_qids), 1, ), )
        if token.shift != 0:
            return 0
        try:
            index = wrt_qids.index(token.qid, )
            diff = _np.zeros((len(wrt_qids), 1, ), )
            diff[index] = 1
            return diff
        except:
            return 0

    #]

