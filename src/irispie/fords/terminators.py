"""
First-order terminators
========================

"""


#[
from __future__ import annotations

from types import (SimpleNamespace, )
import itertools as _it
import numpy as _np
import scipy as _sp
import warnings as _wa

from ..incidences import main as _incidences
from ..incidences.main import Token
from .. import equations as _equations
from ..equations import Equation
from . import simulators as _simulators
from ..aldi.maps import ArrayMap

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Callable, )
    from collections.abc import (Iterable, Sequence, )
    from .simulators import FordSimulatableProtocol
#]


class Terminator:
    """
    First-order terminator class
    """
    #[

    def __init__(
        self,
        simulatable_v: FordSimulatableProtocol,
        columns_simulated: Sequence[int],
        wrt_equations: Iterable[Equation],
    ) -> None:
        """
        """
        last_simulation = columns_simulated[-1]
        first_terminal = last_simulation + 1
        max_lead = simulatable_v.max_lead
        terminal_columns = tuple(range(first_terminal, first_terminal + max_lead, ), )
        num_terminal_columns = len(terminal_columns)
        #
        solution = simulatable_v._gets_solution(deviation=False, )
        vec = simulatable_v._get_dynamic_solution_vectors()
        qid_to_logly = simulatable_v.create_qid_to_logly()
        #
        curr_xi_qids, curr_xi_indexes = vec.get_curr_transition_indexes()
        num_curr_xi_qids = len(curr_xi_qids)
        #
        tokens_from_equations = (
            i for i in _equations.generate_all_tokens_from_equations(wrt_equations, )
            if i.qid in curr_xi_qids
        )
        curr_xi_qid_to_max_shift = _incidences.get_some_shift_by_quantities(tokens_from_equations, max, )
        to_be_zipped = (
            (inx, Token(qid, col, ), )
            for inx, (col, qid, ) in enumerate(_it.product(terminal_columns, curr_xi_qids, ))
            if col <= last_simulation + curr_xi_qid_to_max_shift[qid]
        )
        terminal_column_index, terminal_wrt_spots = zip(*to_be_zipped)
        #
        #
        T = solution.T
        K = solution.K
        #
        # curr_TT := [ T; T @ T; ... ] with current dated rows only
        # curr_KK := [ K; T @ K + K; ... ] with current dated rows only
        curr_TT = []
        curr_KK = []
        cum_T = _np.eye(*T.shape, dtype=T.dtype, )
        cum_K = _np.zeros(K.shape, dtype=K.dtype, )
        for i in range(max_lead, ):
            cum_T = T @ cum_T
            cum_K = T @ cum_K + K
            curr_TT.append(cum_T[curr_xi_indexes, ...])
            curr_KK.append(cum_K[curr_xi_indexes, ...])
        #
        self.terminal_wrt_spots = terminal_wrt_spots
        self._transition_vector = vec.transition_variables
        self._logly_rows = tuple( qid for qid, status in qid_to_logly.items() if status )
        self._first_terminal = first_terminal
        self._max_lead = max_lead
        self._curr_xi_qids = curr_xi_qids
        self._curr_TT = _np.vstack(curr_TT, )
        self._curr_KK = _np.hstack(curr_KK, )
        self._terminal_columns = terminal_columns
        self._terminal_column_index = terminal_column_index
        self._terminit_spots = tuple( Token(i.qid, last_simulation + i.shift, ) for i in vec.transition_variables )
        self._num_terminal_wrt_spots = len(terminal_wrt_spots)
        self.terminal_jacobian_map = None
        self._terminal_jacobian_map_completed = False

    def create_terminal_jacobian_map(self, wrt_spots: Iterable[Token], ) -> None:
        """
        """
        lhs_columns = []
        rhs_columns = []
        wrt_spot_to_index = { spot: inx for inx, spot in enumerate(wrt_spots, ) }
        for rhs_column, spot in enumerate(self._terminit_spots, ):
            try:
                index = wrt_spot_to_index[spot]
            except KeyError:
                continue
            lhs_columns.append(index)
            rhs_columns.append(rhs_column)
        self.terminal_jacobian_map = ArrayMap(
            lhs=([], lhs_columns),
            rhs=([], rhs_columns),
        )

    def terminate_simulation(self, data_array: _np.ndarray, /, ) -> None:
        """
        """
        logly_rows = self._logly_rows
        first_terminal = self._first_terminal
        transition_vector = self._transition_vector
        max_lead = self._max_lead
        terminal_columns = self._terminal_columns
        curr_xi_qids = self._curr_xi_qids
        curr_TT = self._curr_TT
        curr_KK = self._curr_KK
        #
        data_array[logly_rows, :] = _np.log(data_array[logly_rows, :])
        terminit_xi = _simulators.get_init_xi(data_array, transition_vector, first_terminal, )
        terminal_curr_xi_flat = curr_TT @ terminit_xi + curr_KK
        terminal_curr_xi = terminal_curr_xi_flat.reshape(-1, max_lead, order="F", )
        data_array[_np.ix_(curr_xi_qids, terminal_columns)] = terminal_curr_xi
        data_array[logly_rows, :] = _np.exp(data_array[logly_rows, :])

    def terminate_jacobian(self, jacobian_outcome: _np.ndarray, /, ) -> _np.ndarray:
        """
        """
        num_terminal_wrt_spots = len(self.terminal_wrt_spots)
        terminal_column_index = self._terminal_column_index
        #
        terminal_jacobian_outcome = jacobian_outcome[:, -num_terminal_wrt_spots:]
        regular_jacobian_outcome = jacobian_outcome[:,:-num_terminal_wrt_spots].copy()

        if not self._terminal_jacobian_map_completed:
            coo = terminal_jacobian_outcome.tocoo()
            nnz_rows = sorted(set(coo.row))
            _complete_terminal_jacobian_map(self.terminal_jacobian_map, nnz_rows, )
            self._terminal_jacobian_map_completed = True

        curr_TT = self._curr_TT[terminal_column_index, :]
        add_to_regular_jacobian_outcome = terminal_jacobian_outcome @ curr_TT

        _wa.filterwarnings("ignore", category=_sp.sparse.SparseEfficiencyWarning, )
        regular_jacobian_outcome[self.terminal_jacobian_map.lhs] += add_to_regular_jacobian_outcome[self.terminal_jacobian_map.rhs]
        _wa.filterwarnings("default", category=_sp.sparse.SparseEfficiencyWarning, )

        return regular_jacobian_outcome

    #]


def _complete_terminal_jacobian_map(
    self: ArrayMap,
    nnz_rows: Iterable[int],
    /,
) -> None:
    """
    """
    lhs_columns = self.lhs[1]
    rhs_columns = self.rhs[1]
    lhs_grid = _np.ix_(nnz_rows, lhs_columns, )
    rhs_grid = _np.ix_(nnz_rows, rhs_columns, )
    self.lhs = lhs_grid
    self.rhs = rhs_grid

