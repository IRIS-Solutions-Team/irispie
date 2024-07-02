"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

import numpy as _np

from ..incidences import main as _incidence
from ..aldi.maps import (ArrayMap, )
from ..equations import (Equation, )

from . import base

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, )
    from collections.abc import (Collection, Iterable, )
#]


class _SteadyJacobian(base.Jacobian, ):
    """
    """
    #[

    _create_map = staticmethod(ArrayMap.static)
    _num_diff_columns = ...

    @staticmethod
    def _calculate_shape(
        eids: Collection[int],
        wrt_something: Collection[Any],
        num_columns_to_eval: int,
    ) -> tuple[int, int]:
        """
        """
        return len(eids), len(wrt_something),

    def _create_eid_to_wrts(
        self,
        equations: Iterable[Equation],
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
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, column_offset, )
        return self._create_jacobian_matrix(diff_array, )

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
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, column_offset, )
        A = self._create_jacobian_matrix(diff_array[:, 0:1], )
        B = self._create_jacobian_matrix(diff_array[:, 1:2], )
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

