"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, TypeAlias, Literal, )
import numpy as _np

from ..dataslates import main as _dataslates
from ..plans import main as _plans
from ..fords import solutions as _solutions
from ..fords import descriptors as _descriptors
#]


def simulate_flat(
    solution: _solutions.Solution,
    solution_vectors: _descriptors.SolutionVectors,
    dataslate: _dataslates.HorizontalDataslate,
    vid: int,
    /,
    *,
    plan: _plans.PlanSimulate | None = None,
    deviation: bool = False,
) -> None:
    """
    """
    #[
    columns_to_run = dataslate.base_columns
    boolex_logly = dataslate.boolex_logly
    working_data = dataslate.get_data_variant(0, )
    #
    column_start = columns_to_run[0]
    column_slice = slice(column_start, column_start+len(columns_to_run))
    column_array = _np.array(range(column_start, column_start+len(columns_to_run)))
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
    if any(boolex_logly):
        working_data[boolex_logly, :] = _np.log(working_data[boolex_logly, :])
    #
    curr_state = _dataslates.retrieve_vector_from_data_array(
        working_data, vec.transition_variables, column_start-1,
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
    unanticipated_shocks = _extract_shock_values(working_data, vec.unanticipated_shocks, )
    anticipated_shocks = _extract_shock_values(working_data, vec.anticipated_shocks, )
    measurement_shocks = _extract_shock_values(working_data, vec.measurement_shocks, )
    #
    Rx = [R]
    forward = None
    #
    anticipated_shock_column_incidence = list(_np.any(anticipated_shocks[:, column_array] != 0, axis=0))
    if any(anticipated_shock_column_incidence):
        # Last shock at t+forward
        forward = len(anticipated_shock_column_incidence) - anticipated_shock_column_incidence[::-1].index(True) - 1
        Rk = solution.expand_square_solution(forward, )
        Rx = Rx + (Rk if Rk is not None else [])
    #
    shock_column_end = (
        column_start + forward
        if forward is not None
        else column_start - 1
    )
    #
    for t in column_array:
        anticipated_shock_impact = sum(
            Rx[k] @ anticipated_shocks[:, s] if Rx[k] is not None else 0
            for k, s in enumerate(range(t, shock_column_end+1))
        )
        curr_state = T @ curr_state + P @ unanticipated_shocks[:, t] + anticipated_shock_impact + K
        working_data[no_shift_state_to_slab_lhs, t] = curr_state[no_shift_state_to_slab_rhs]
        #
        y = Z @ curr_state + H @ measurement_shocks[:, t] + D
        working_data[measurement_variables_index, t] = y
        #
    _store_shock_values(unanticipated_shocks, working_data, vec.unanticipated_shocks, )
    _store_shock_values(anticipated_shocks, working_data, vec.anticipated_shocks, )
    _store_shock_values(measurement_shocks, working_data, vec.measurement_shocks, )
    #
    if any(boolex_logly):
        working_data[boolex_logly, :] = _np.exp(working_data[boolex_logly, :])
    #]


def _extract_shock_values(working_data, shock_vector, ) -> _np.ndarray:
    """
    """
    #[
    shock_qids = [t.qid for t in shock_vector]
    shock_values = working_data[shock_qids, :]
    _np.nan_to_num(shock_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0, )
    return shock_values
    #]


def _store_shock_values(shock_values, working_data, shock_vector, ) -> None:
    """
    """
    #[
    shock_qids = [t.qid for t in shock_vector]
    working_data[shock_qids, :] = shock_values
    #]

