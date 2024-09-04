"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

import numpy as _np
from types import (SimpleNamespace, )

from ..incidences.main import (Token, )
from ..incidences import main as _incidence
from ..aldi.maps import (ArrayMap, )
from ..equations import (Equation, )
from ..jacobians.base import (DenseJacobian, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, )
    from collections.abc import (Collection, Iterable, )
    from ..aldi.differentiators import (AtomFactoryProtocol, )
#]


# Implement AtomFactoryProtocol


def _flat_create_diff_for_token(
    token: Token,
    wrt_qids: tuple[int],
    /,
) -> _np.ndarray | int:
    """
    """
    try:
        index = wrt_qids.index(token.qid, )
    except ValueError:
        return 0
    diff = _np.zeros((len(wrt_qids), 1, ))
    diff[index, 0] = 1
    return diff


_FLAT_ATOM_FACTORY = SimpleNamespace(
    create_data_index_for_token=lambda token: (token.qid, token.shift, ),
    create_diff_for_token=_flat_create_diff_for_token,
    get_diff_shape=lambda wrt_qids: (len(wrt_qids), 1, ),
)


def _nonflat_create_diff_for_token(
    token: Token,
    wrt_qids: tuple[int],
    /,
) -> _np.ndarray | int:
    """
    """
    try:
        index = wrt_qids.index(token.qid, )
    except ValueError:
        return 0
    diff = _np.zeros((len(wrt_qids), 2, ))
    diff[index, :] = 1, token.shift,
    return diff


_NONFLAT_ATOM_FACTORY = SimpleNamespace(
    create_data_index_for_token=lambda token: (token.qid, token.shift, ),
    create_diff_for_token=_nonflat_create_diff_for_token,
    get_diff_shape=lambda wrt_qids: (len(wrt_qids), 2, ),
)



class _SteadyJacobian(DenseJacobian, ):
    """
    """
    #[

    def _populate_map(self, *args, **kwargs, ) -> None:
        """
        """
        self._map = ArrayMap.static(*args, **kwargs, )

    def eval(*args, **kwargs, ): raise NotImplementedError

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

        #]


class FlatSteadyJacobian(_SteadyJacobian, ):
    """
    """
    #[

    _atom_factory = _FLAT_ATOM_FACTORY

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

    #]


class NonflatSteadyJacobian(_SteadyJacobian, ):
    """
    """
    #[

    # Assigned in the evaluator
    NONFLAT_STEADY_SHIFT = ...

    _atom_factory = _NONFLAT_ATOM_FACTORY

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
        k = self.NONFLAT_STEADY_SHIFT
        return _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))

    #]

