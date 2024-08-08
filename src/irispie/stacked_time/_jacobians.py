"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from types import (SimpleNamespace, )
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


class Jacobian(base.Jacobian, ):
    """
    """
    #[

    def _populate_map(
        self,
        eids: list[int],
        eid_to_wrt_tokens: dict[int, Iterable[Token]],
        tokens_in_columns_on_lhs: list[Token],
        eid_to_rhs_offset: dict[int, int],
        **kwargs,
    ) -> None:
        """
        """
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
                for rhs_column in self._columns_to_eval:
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
        self._map = map

    def eval(
        self,
        data_array: _np.ndarray,
        column_offset: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(data_array, column_offset, )
        extended_jacobian_matrix = self._create_jacobian_matrix(diff_array, )
        jacobian_matrix = extended_jacobian_matrix[:, :42]
        for i in range(8):
            jacobian_matrix += 0.*extended_jacobian_matrix[:, 42+i*42:42+i*42+42]
        # j = jacobian_matrix
        # x = extended_jacobian_matrix
        # import IPython; IPython.embed(header=__name__)
        return jacobian_matrix

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

    @property
    def _atom_factory(self, /, ) -> AtomFactoryProtocol:
        """
        """
        columns_to_eval = self._columns_to_eval
        num_columns_to_eval = self._num_columns_to_eval
        #
        def create_diff_for_token(
            self,
            token: Token,
            wrt_tokens: tuple[Token],
        ) -> _np.ndarray | int:
            """
            """
            num_wrts = len(wrt_tokens)
            try:
                index = wrt_tokens.index(token, )
            except ValueError:
                return 0
            diff = _np.zeros((num_wrts, self._num_columns_to_eval, ), dtype=_np.float64, )
            diff[index] = 1
            return diff
        #
        return SimpleNamespace(
            create_data_index_for_token=lambda token: (token.qid, token.shift + _np.array(columns_to_eval, ), ),
            create_diff_for_token=create_diff_for_token,
            get_diff_shape=lambda wrt_tokens: (len(wrt_tokens), num_columns_to_eval, ),
        )

    #]

