"""
First-order terminators
========================

"""


#[
from __future__ import annotations

from types import (SimpleNamespace, )
import numpy as _np

from . import simulators as _simulators

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Callable, )
    from .simulators import FordSimulatableProtocol
#]


def create_terminators(
    simulatable_v: FordSimulatableProtocol,
    terminal: Literal["data", "first_order", ],
) -> SimpleNamespace:
    """
    """
    #[
    max_lead = simulatable_v.max_lead
    if terminal != "first_order" or not max_lead:
        return SimpleNamespace(
            terminate_simulation=None,
            terminate_jacobian=None,
        )
    #
    import ipdb; ipdb.set_trace()
    solution = simulatable_v.get_singleton_solution(deviation=False, )
    vec = simulatable_v.solution_vectors
    curr_xi_qids, curr_xi_indexes = vec.get_curr_transition_indexes()
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
    curr_TT = _np.vstack(curr_TT, )
    curr_KK = _np.hstack(curr_KK, )
    #
    qid_to_logly = simulatable_v.create_qid_to_logly()
    logly_rows = tuple( qid for qid, status in qid_to_logly.items() if status )
    #
    #
    def terminate_simulation(
        data_array: _np.ndarray,
        last_simulation: int,
    ) -> None:
        """
        """
        first_terminal = last_simulation + 1
        terminal_columns = tuple(range(first_terminal, first_terminal + max_lead, ), )
        data_array[logly_rows, :] = _np.log(data_array[logly_rows, :])
        terminit_xi = _simulators.get_init_xi(data_array, first_terminal, vec, )
        #
        terminal_curr_xi = curr_TT @ terminit_xi + curr_KK
        #
        terminal_curr_xi = terminal_curr_xi.reshape(-1, max_lead, order="F", )
        data_array[_np.ix_(curr_xi_qids, terminal_columns)] = terminal_curr_xi
        data_array[logly_rows, :] = _np.exp(data_array[logly_rows, :])
    #
    #
    def terminate_jacobian(
    ) -> None:
        """
        """
        pass
    #
    #
    return SimpleNamespace(
        terminate_simulation=terminate_simulation,
        terminate_jacobian=terminate_jacobian,
    )
    #]


