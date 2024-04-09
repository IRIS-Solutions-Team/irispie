"""
Time frames for dynamic simulations
"""


#[
from __future__ import annotations

from typing import (NamedTuple, )
import numpy as _np

from .dates import (Period, )
from .dataslates.main import (Dataslate, )
from .plans.main import (SimulationPlan, )
#]


class Frame(NamedTuple, ):
    """
    """
    #[

    start: Period
    end: Period
    simulation_end: Period | None = None,

    #]


def split_into_frames(
    model,
    dataslate: Dataslate | None,
    plan: SimulationPlan | None,
) -> tuple[Frame]:
    """
    """
    if not model.is_singleton:
        raise ValueError("Model must be a singleton")
    if not dataslate.is_singleton:
        raise ValueError("Dataslate must be a singleton")
    breakpoints = _get_breakpoints(model, dataslate, plan, )
    return breakpoints


def _get_breakpoints(
    model,
    dataslate: Dataslate | None,
    plan: SimulationPlan | None,
) -> tuple[Frame]:
    """
    """
    plannable = model.get_simulation_plannable()
    can_be_endogenized = plannable.can_be_endogenized_unanticipated
    can_be_exogenized = plannable.can_be_exogenized_unanticipated
    name_to_row = dataslate.create_name_to_row()
    rows = tuple(name_to_row[name] for name in can_be_endogenized)
    periods = dataslate.base_periods
    columns = dataslate.base_columns
    breakpoints = _np.zeros((len(columns), ), dtype=bool, )
    breakpoints[0] = True
    if rows and columns:
        cutout = dataslate.get_data_variant(0, )[rows, :][:, columns]
        breakpoints = _update_breakpoints(breakpoints, cutout, )
    #
    if plan is not None:
        cutout = plan.tabulate_registered_points(
            "endogenized_unanticipated", can_be_endogenized, periods,
        )
        breakpoints = _update_breakpoints(breakpoints, cutout, )
        cutout = plan.tabulate_registered_points(
            "exogenized_unanticipated", can_be_exogenized, periods,
        )
        breakpoints = _update_breakpoints(breakpoints, cutout, )
    return breakpoints


def _update_breakpoints(breakpoints, new_array, ):
    """
    """
    if new_array is None:
        return breakpoints
    new_breakpoints = _np.any(_np.isfinite(new_array) & (new_array != 0), axis=0, )
    return breakpoints | new_breakpoints

