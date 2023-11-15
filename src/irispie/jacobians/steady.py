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


class _SteadyJacobian(_base.Jacobian, ):
    """
    """
    #[

    _num_diff_columns = ...

    def _create_eid_to_wrts(
        self,
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
        wrt_qids: dict[int, tuple[int]],
        /,
    ) -> _np.ndarray:
        """
        """
        if token is None:
            return _np.zeros((len(wrt_qids), self._num_diff_columns, ))
        try:
            index = wrt_qids.index(token.qid)
        except:
            index = None
        if index is None:
            return 0
        diff = _np.zeros((len(wrt_qids), self._num_diff_columns, ))
        diff[index, :] = self._create_diff_value_for_token(token, )
        return diff

    def _create_diff_value_for_token(
        self,
        token: _incidence.Token,
        /,
    ) -> _np.ndarray:
        """
        """
        ...

        #]


class FlatSteadyJacobian(_SteadyJacobian, ):
    """
    """
    #[

    _num_diff_columns = 1

    def eval(
        self,
        steady_array: _np.ndarray,
        column_offset: int | None,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(
            steady_array, column_offset, steady_array,
        )
        return self._create_jacobian(self._shape, diff_array, self._map, )

    def _create_diff_value_for_token(
        self,
        token: _incidence.Token,
        /,
    ) -> _np.ndarray:
        """
        """
        return 1

    #]


class NonflatSteadyJacobian(_SteadyJacobian, ):
    """
    """
    #[

    _num_diff_columns = 2
    nonflat_steady_shift = None

    def eval(
        self,
        steady_array: _np.ndarray,
        column_offset: int,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(
            steady_array, column_offset, steady_array,
        )
        A = self._create_jacobian(self._shape, diff_array[:, 0:1], self._map, )
        B = self._create_jacobian(self._shape, diff_array[:, 1:2], self._map, )
        k = self.nonflat_steady_shift
        j = _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))
        return _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))

    def _create_diff_value_for_token(
        self,
        token: _incidence.Token,
        /,
    ) -> _np.ndarray:
        """
        """
        return 1, token.shift

    #]

