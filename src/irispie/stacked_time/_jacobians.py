"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

import numpy as _np


from ..jacobians import base
from ..aldi.maps import (ArrayMap, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from types import (EllipsisType, )
    from collections.abc import (Iterable, )
    from ..incidences.main import (Token, )
    from ..equations import (Equation, )
#]


def _create_map(
    eids: list[int],
    eid_to_wrt_tokens: dict[int, Any],
    tokens_in_columns_on_lhs: list[Any],
    eid_to_rhs_offset: dict[int, int],
    *,
    num_columns_to_eval: int,
    **kwargs,
) -> Self:
    """
    """
    #[
    map = ArrayMap()
    token_to_lhs_column = {
        t: i
        for i, t in enumerate(tokens_in_columns_on_lhs, )
    }
    num_eids = len(eids)
    #
    map_tuples = []
    rhs_row_offset = 0
    for eqn_enum, eid in enumerate(eids, ):
        wrt_tokens = eid_to_wrt_tokens[eid]
        for rhs_row, tok in enumerate(wrt_tokens, start=rhs_row_offset, ):
            for rhs_column in range(num_columns_to_eval, ):
                try:
                    tok_shifted = tok.shifted(rhs_column, )
                    lhs_column = token_to_lhs_column[tok_shifted]
                    lhs_row = eqn_enum + num_eids*rhs_column
                    map_tuples.append((lhs_row, lhs_column, rhs_row, rhs_column, ))
                except KeyError:
                    continue
        rhs_row_offset += len(wrt_tokens)
    #
    lhs_rows, lhs_columns, rhs_rows, rhs_columns = zip(*map_tuples, )
    map.lhs = (list(lhs_rows), list(lhs_columns), )
    # map.lhs = (list(lhs_columns), list(lhs_rows), )
    map.rhs = (list(rhs_rows), list(rhs_columns), )
    return map
    #]


class Jacobian(base.Jacobian, ):
    """
    """
    #[

    _create_map = staticmethod(_create_map)

    @staticmethod
    def _calculate_shape(
        eids: Collection[int],
        wrt_something: Collection[Any],
        num_columns_to_eval: int,
    ) -> tuple[int, int]:
        """
        """
        return len(eids)*num_columns_to_eval, len(wrt_something),

    def eval(
        self,
        data_array: _np.ndarray,
        column_offset: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(data_array, column_offset, )
        return self._create_jacobian_matrix(diff_array, )

    def _create_eid_to_wrts(
        self,
        equations: Iterable[Equation],
        all_wrt_tokens: Iterable[int],
        /,
    ) -> dict[int, tuple[int, ...]]:
        """
        """
        all_wrt_qids = set(tok.qid for tok in all_wrt_tokens)
        return {
            eqn.id: tuple(
                tok for tok in eqn.incidence
                if tok.qid in all_wrt_qids
            )
            for eqn in equations
        }

    # ===== Implement AtomFactoryProtocol =====

    def create_data_index_for_token(
        self,
        token: Token,
    ) -> tuple[int, _np.ndarray]:
        """
        """
        if not hasattr(self, "_column_indexes", ):
            self._column_indexes = _np.arange(self.num_columns_to_eval, )
        return token.qid, token.shift + self._column_indexes,

    def create_diff_for_token(
        self,
        token: Token,
        wrt_tokens: tuple[Token],
    ) -> _np.ndarray | int:
        """
        """
        num_wrts = len(wrt_tokens)
        if token is None:
            return _np.zeros((num_wrts, self.num_columns_to_eval, ), dtype=_np.float64, )
        try:
            index = wrt_tokens.index(token, )
            diff = _np.zeros((num_wrts, self.num_columns_to_eval, ), dtype=_np.float64, )
            diff[index] = 1
            return diff
        except:
            return 0

    #]

