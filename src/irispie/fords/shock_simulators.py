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
    if model_v.num_variants != 1:
        raise ValueError("Simulator requires a singleton simulatable object")
    #
    solution = model_v.get_solution()
    solution_vectors = model_v.solution_vectors
    base_columns = dataslate_v.base_columns
    first_column = frame.start - dataslate_v.periods[0]
    last_column = frame.simulation_end - dataslate_v.periods[0]
    columns_to_run = tuple(range(first_column, last_column+1))
    working_data = dataslate_v.get_data_variant(0, )
    anticipated_shocks = extract_shock_values(working_data, solution_vectors.anticipated_shocks, )
    #
    impact = [None] * dataslate_v.num_periods
    column_incidence = _np.any(anticipated_shocks[:, columns_to_run], axis=0, ).tolist()
    if any(column_incidence, ):
        #
        # Last shock is at t+forward
        forward = len(column_incidence) - column_incidence[::-1].index(True, ) - 1
        Rx = get_solution_expansion(solution, forward, )
        #
        # Last column within dataslate_v
        last_ant_column = (
            first_column + forward
            if forward is not None
            else first_column - 1
        )
        #
        for t in columns_to_run:
            impact[t] = sum(
                Rx[k] @ anticipated_shocks[:, s]
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

