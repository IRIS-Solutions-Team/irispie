"""
"""


#[
from __future__ import annotations

from typing import (Callable, )
import numpy as _np
import functools as _ft

from ..frames import (Frame, )
from ..dataslates.main import (Dataslate, )
from .solutions import (Solution, )
#]


def _simulate_anticipated_shocks(
    model_v,
    dataslate_v: Dataslate,
    frame: Frame,
    get_solution_expansion: Callable,
) -> list[_np.ndarray | None]:
    """
    """
    #[
    #
    # Preallocate impact with None, and return immediately if no anticipated
    # shocks are present
    impact = [None] * dataslate_v.num_periods
    vec = model_v._get_dynamic_solution_vectors()
    if not vec.anticipated_shocks:
        return impact
    #
    # Deviation is irrelevant for the impact of anticipated shocks
    solution = model_v._gets_solution()
    base_columns = dataslate_v.base_columns
    first_column = frame.start - dataslate_v.periods[0]
    last_column = frame.simulation_end - dataslate_v.periods[0]
    columns_to_run = tuple(range(first_column, last_column+1))
    working_data = dataslate_v.get_data_variant(0, )
    #
    v_array = extract_shock_values(
        working_data,
        vec.anticipated_shocks,
    )
    #
    column_incidence = _np.any(v_array[:, columns_to_run], axis=0, ).tolist()
    if any(column_incidence, ):
        #
        # Last shock is at t+forward
        last_index_reversed = column_incidence[::-1].index(True)
        forward = len(column_incidence) - last_index_reversed - 1
        Rx = get_solution_expansion(solution, forward, )
        #
        # Last column within dataslate_v
        last_ant_column = first_column + forward
        #
        for t in columns_to_run:
            impact[t] = sum(
                Rx[k] @ v_array[:, s]
                for k, s in enumerate(range(t, last_ant_column+1))
            )
    return impact
    #]


simulate_square_anticipated_shocks = _ft.partial(
    _simulate_anticipated_shocks,
    get_solution_expansion=Solution.expand_square_solution,
)


simulate_triangular_anticipated_shocks = _ft.partial(
    _simulate_anticipated_shocks,
    get_solution_expansion=Solution.expand_triangular_solution,
)


def extract_shock_values(working_data, shock_vector, ) -> _np.ndarray:
    """
    """
    #[
    shock_qids = [t.qid for t in shock_vector]
    return working_data[shock_qids, :]
    #]

