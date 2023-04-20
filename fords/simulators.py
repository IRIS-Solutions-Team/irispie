"""
First-order system simulators
"""

#[
from __future__ import annotations
# from IPython import embed

from typing import (Self, TypeAlias, NoReturn, Literal, )
import numpy as np_

from ..fords import (solutions as sl_, descriptors as de_, )
from ..dataman import (databanks as db_, )
#]


def simulate_flat(
    solution: sl_.Solution,
    solution_vectors: de_.SolutionVectors,
    data: np_.ndarray,
    columns_to_run: list[int],
    deviation: bool,
    anticipate: bool,
):
    column_start = columns_to_run[0]
    column_slice = slice(column_start, column_start+len(columns_to_run))
    column_array = np_.array(range(column_start, column_start+len(columns_to_run)))

    vec = solution_vectors

    T = solution.T
    R = solution.R
    K = solution.K if not deviation else 0

    Z = solution.Z
    H = solution.H
    D = solution.D if not deviation else 0

    curr_state = np_.array([
        data[t.qid, column_start-1+t.shift] 
        for t in vec.transition_variables
    ], dtype=float).reshape(-1, 1)

    missing_initials = np_.isnan(curr_state)
    unnecessary_initials = ~np_.array(vec.initial_conditions).reshape(-1, 1)
    missing_unnecessary_initials = missing_initials & unnecessary_initials
    curr_state[missing_unnecessary_initials] = 0

    no_shift_state_to_slab_lhs = [
        t.qid
        for t in vec.transition_variables
        if t.shift == 0
    ]

    no_shift_state_to_slab_rhs = [
        j 
        for j, t in enumerate(vec.transition_variables)
        if t.shift == 0
    ]

    measurement_variables_to_slab = [t.qid for t in vec.measurement_variables]

    transition_shocks_in_slab = [t.qid for t in vec.transition_shocks]
    transition_shocks = data[transition_shocks_in_slab, :]

    measurement_shocks_in_slab = [t.qid for t in vec.measurement_shocks]
    measurement_shocks = data[[t.qid for t in vec.measurement_shocks], :]

    transition_shocks[np_.isnan(transition_shocks)] = 0
    measurement_shocks[np_.isnan(measurement_shocks)] = 0

    Rx = [R]
    forward = -1

    shock_column_incidence = list(np_.any(transition_shocks[:, column_array] != 0, axis=0))
    if any(shock_column_incidence):
        # Last shock at t+forward
        forward = len(shock_column_incidence) - shock_column_incidence[::-1].index(True) - 1
        if anticipate:
            Rk = solution.expand_square_solution(forward)
        else:
            Rk = [None]*forward
        Rx = Rx + (Rk if Rk is not None else [])

    shock_column_end = column_start + forward
    for t in column_array:
        shock_impact = sum( 
            Rx[k] @ transition_shocks[:, (s,)] if Rx[k] is not None else 0 
            for k, s in enumerate(range(t, shock_column_end+1))
        )
        curr_state = T @ curr_state + shock_impact + K
        data[no_shift_state_to_slab_lhs, t] = curr_state[no_shift_state_to_slab_rhs].flat

        y = Z @ curr_state + H @ measurement_shocks[:, (t,)] + D
        data[measurement_variables_to_slab, t] = y.flat

    data[transition_shocks_in_slab, :] = transition_shocks
    data[measurement_shocks_in_slab, :] = measurement_shocks

    return data

