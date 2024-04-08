"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, Callable, )
import numpy as _np
import functools as _ft
import wlogging as _wl

from ..dataslates import main as _dataslates
from ..plans import main as _plans
from ..fords import solutions as _solutions
from ..fords import descriptors as _descriptors
#]


class _FordSimulatableProtocol:
    """
    """
    #[

    num_variants: int

    solution_vectors: _descriptors.SolutionVectors

    def get_solution(self, /, ) -> _solutions.Solution: ...

    #]


def simulate(
    simulatable_v: _FordSimulatableProtocol,
    dataslate_v: _dataslates.Dataslate,
    vid: int,
    logger: _wl.Logger,
    /,
    *,
    plan: _plans.PlanSimulate | None = None,
    deviation: bool = False,
) -> dict[str, Any]:
    return simulate_flat(
        simulatable_v, dataslate_v, vid,
        plan=plan,
        deviation=deviation,
    )


def simulate_flat(
    simulatable_v: _FordSimulatableProtocol,
    dataslate_v: _dataslates.Dataslate,
    vid: int,
    /,
    *,
    plan: _plans.PlanSimulate | None = None,
    deviation: bool = False,
) -> None:
    """
    """
    #[
    if simulatable_v.num_variants != 1:
        raise ValueError("Simulator requires a singleton simulatable object")
    #
    solution = simulatable_v.get_solution()
    solution_vectors = simulatable_v.solution_vectors
    columns_to_run = dataslate_v.base_columns
    logly_indexes = dataslate_v.logly_indexes
    working_data = dataslate_v.get_data_variant(0, )
    #
    first_column = columns_to_run[0]
    last_column = columns_to_run[-1]
    column_range = tuple(range(first_column, last_column+1))
    #
    vec = solution_vectors
    #
    T = solution.T
    P = solution.P
    R = solution.R
    K = solution.K if not deviation else 0
    #
    Z = solution.Z
    H = solution.H
    D = solution.D if not deviation else 0
    #
    if logly_indexes:
        working_data[logly_indexes, ...] = _np.log(working_data[logly_indexes, ...])
    #
    curr_state = _dataslates.retrieve_vector_from_data_array(
        working_data, vec.transition_variables, first_column-1,
    )
    #
    missing_initials = _np.isnan(curr_state, )
    unnecessary_initials = ~_np.array(vec.are_initial_conditions, )
    missing_unnecessary_initials = missing_initials & unnecessary_initials
    curr_state[missing_unnecessary_initials] = 0
    #
    no_shift_state_to_slab_lhs = [
        t.qid
        for t in vec.transition_variables
        if t.shift == 0
    ]
    #
    no_shift_state_to_slab_rhs = [
        j 
        for j, t in enumerate(vec.transition_variables)
        if t.shift == 0
    ]
    #
    measurement_variables_index = [t.qid for t in vec.measurement_variables]
    #
    u = _extract_shock_values(working_data, vec.unanticipated_shocks, )
    w = _extract_shock_values(working_data, vec.measurement_shocks, )
    #
    v_impact \
        = simulate_square_anticipated_shocks(simulatable_v, dataslate_v, )
    #
    for t in column_range:
        curr_state = T @ curr_state + P @ u[:, t] + K
        if v_impact[t] is not None:
            curr_state += v_impact[t]
        #
        working_data[no_shift_state_to_slab_lhs, t] = curr_state[no_shift_state_to_slab_rhs]
        #
        y = Z @ curr_state + H @ w[:, t] + D
        working_data[measurement_variables_index, t] = y
        #
    if logly_indexes:
        working_data[logly_indexes, ...] = _np.exp(working_data[logly_indexes, ...])
    #
    info = {
        "method": "first_order",
        "anticipatory_shocks_impact": v_impact,
    }
    #
    return info
    #]


def _extract_shock_values(working_data, shock_vector, ) -> _np.ndarray:
    """
    """
    #[
    shock_qids = [t.qid for t in shock_vector]
    shock_values = working_data[shock_qids, :]
    return shock_values
    #]


def _simulate_anticipated_shocks(
    simulatable_v: _FordSimulatableProtocol,
    dataslate_v: _dataslates.Dataslate,
    expand_solution: Callable,
) -> list[_np.ndarray | None]:
    """
    """
    #[
    if simulatable_v.num_variants != 1:
        raise ValueError("Simulator requires a singleton simulatable object")
    #
    solution = simulatable_v.get_solution()
    solution_vectors = simulatable_v.solution_vectors
    base_columns = dataslate_v.base_columns
    first_column = base_columns[0]
    last_column = base_columns[-1]
    column_range = tuple(range(first_column, last_column+1))
    working_data = dataslate_v.get_data_variant(0, )
    anticipated_shocks = _extract_shock_values(working_data, solution_vectors.anticipated_shocks, )
    #
    impact = [None] * dataslate_v.num_periods
    column_incidence = _np.any(anticipated_shocks[:, column_range], axis=0, ).tolist()
    if any(column_incidence, ):
        #
        # Last shock is at t+forward
        forward = len(column_incidence) - column_incidence[::-1].index(True, ) - 1
        Rx = expand_solution(solution, forward, )
        #
        # Last column within dataslate_v
        last_ant_column = (
            first_column + forward
            if forward is not None
            else first_column - 1
        )
        #
        for t in column_range:
            impact[t] = sum(
                Rx[k] @ anticipated_shocks[:, s]
                for k, s in enumerate(range(t, last_ant_column+1))
            )
    return impact
    #]


simulate_square_anticipated_shocks = _ft.partial(
    _simulate_anticipated_shocks,
    expand_solution=_solutions.Solution.expand_square_solution,
)


simulate_triangular_anticipated_shocks = _ft.partial(
    _simulate_anticipated_shocks,
    expand_solution=_solutions.Solution.expand_triangular_solution,
)

