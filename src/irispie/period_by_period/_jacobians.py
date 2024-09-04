"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from types import (SimpleNamespace, )
from collections.abc import (Iterable, )
import numpy as _np

from ..incidences.main import (Token, )
from ..incidences import main as _incidence
from ..aldi.maps import (ArrayMap, )
from ..aldi.differentiators import (AtomFactoryProtocol, )

from ..jacobians.base import (DenseJacobian, )
#]


def _create_diff_for_token(
    token: Token,
    wrt_qids: tuple[int],
    /,
) -> _np.ndarray | int:
    """
    """
    if token.shift != 0:
        return 0
    try:
        index = wrt_qids.index(token.qid, )
    except ValueError:
        return 0
    diff = _np.zeros((len(wrt_qids), 1, ), )
    diff[index] = 1
    return diff


_ATOM_FACTORY = SimpleNamespace(
    create_data_index_for_token=lambda token: (token.qid, token.shift, ),
    create_diff_for_token=_create_diff_for_token,
    get_diff_shape=lambda wrt_qids: (len(wrt_qids), 1, ),
)


class Jacobian(DenseJacobian, ):
    """
    """
    #[

    _atom_factory: AtomFactoryProtocol = _ATOM_FACTORY

    def _populate_map(self, *args, **kwargs, ) -> None:
        """
        """
        self._map = ArrayMap.static(*args, **kwargs, )

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
        all_wrt_qids: Iterable[Tokens],
        /,
    ) -> dict[int, tuple[Token, ...]]:
        """
        """
        return {
            eqn.id: tuple(
                qid for qid in all_wrt_qids
                if _incidence.is_qid_zero_in_tokens(eqn.incidence, qid, )
            )
            for eqn in equations
        }

    #]


