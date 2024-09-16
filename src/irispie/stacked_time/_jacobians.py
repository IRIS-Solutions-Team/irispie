"""
Jacobians for dynamic period-by-period systems
"""


#[
from __future__ import annotations

from types import (SimpleNamespace, )
import numpy as _np
import itertools as _it

from ..jacobians.base import SparseJacobian, DenseJacobian
from ..fords.terminators import (Terminator, )
from ..aldi.maps import (ArrayMap, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from types import (EllipsisType, )
    from collections.abc import (Iterable, )
    from ..incidences.main import (Token, )
    from ..equations import (Equation, )
#]


class Jacobian(SparseJacobian, ):
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
        token_to_lhs_column = {
            t: i
            for i, t in enumerate(tokens_in_columns_on_lhs, )
        }
        num_eids = len(eids)
        map_tuples = []
        rhs_row_offset = 0
        for eqn_enum, eid in enumerate(eids, ):
            wrt_tokens_in_equation = eid_to_wrt_tokens[eid]
            for rhs_row, tok in enumerate(wrt_tokens_in_equation, start=rhs_row_offset, ):
                for rhs_column, column_to_run in enumerate(self._columns_to_eval, ):
                    try:
                        tok_shifted = tok.shifted(column_to_run, )
                        lhs_column = token_to_lhs_column[tok_shifted]
                        lhs_row = eqn_enum + num_eids*rhs_column
                        map_tuples.append((lhs_row, lhs_column, rhs_row, rhs_column, ))
                    except KeyError:
                        continue
            rhs_row_offset += len(wrt_tokens_in_equation)
        lhs_rows, lhs_columns, rhs_rows, rhs_columns = zip(*map_tuples, )
        self._map = ArrayMap(
            lhs=(list(lhs_rows), list(lhs_columns), ),
            rhs=(list(rhs_rows), list(rhs_columns), ),
        )

    def eval(
        self,
        data_array: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(data_array, )
        return self._create_jacobian_matrix(diff_array, )
        # jacobian_matrix = extended_jacobian_matrix[:, :42]
        # for i in range(8):
        #     jacobian_matrix += 0.*extended_jacobian_matrix[:, 42+i*42:42+i*42+42]
        # # j = jacobian_matrix
        # # x = extended_jacobian_matrix
        # return jacobian_matrix

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
        columns_to_eval = _np.array(self._columns_to_eval, dtype=int, )
        num_columns_to_eval = self._num_columns_to_eval
        #
        def create_data_index_for_token(token: Token, /, ) -> tuple[int, _np.ndarray]:
            """
            """
            return token.qid, token.shift + columns_to_eval, 
        #
        def create_diff_for_token(
            token: Token,
            wrt_tokens_in_equation: tuple[Token, ...],
            /,
        ) -> _np.ndarray | int:
            """
            """
            num_wrts_in_equation = len(wrt_tokens_in_equation)
            try:
                index = wrt_tokens_in_equation.index(token, )
            except ValueError:
                return 0
            diff_shape = (num_wrts_in_equation, self._num_columns_to_eval, )
            diff = _np.zeros(diff_shape, dtype=_np.float64, )
            diff[index, :] = 1
            return diff
        #
        def get_diff_shape(wrt_tokens: tuple[Token, ...], /, ) -> tuple[int, int]:
            """
            """
            return len(wrt_tokens), self._num_columns_to_eval, 
        #
        return SimpleNamespace(
            create_data_index_for_token=create_data_index_for_token,
            create_diff_for_token=create_diff_for_token,
            get_diff_shape=get_diff_shape,
        )

    #]

