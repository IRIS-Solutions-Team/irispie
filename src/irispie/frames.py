"""
Time frames for dynamic simulations
"""


#[
from __future__ import annotations

from typing import (NamedTuple, Callable, Iterable, )
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

    def remove_unanticipated_outside_frame(self, dataslate, model, /, ):
        """
        """
        #[
        if self.start == self.simulation_end:
            return
        plannable = model_v.get_plannable()
        name_to_qid = model_v.create_name_to_qid()
        if not hasattr(plannable, "can_be_endogenized_unanticipated"):
            return
        unanticipated_names = plannable.can_be_endogenized_unanticipated
        unanticipated_qids = tuple(name_to_qid[name] for name in unanticipated_names)
        second_period = self.start + 1
        data = dataslate_f.get_data_variant(0, )
        data[unanticipated_qids, second_period:] = 0

    #]


def split_into_frames(
    model,
    dataslate: Dataslate,
    plan: SimulationPlan | None,
    get_simulation_end: Callable,
) -> tuple[Frame, ...]:
    """
    """
    if not model.is_singleton:
        raise ValueError("Model must be a singleton")
    if not dataslate.is_singleton:
        raise ValueError("Dataslate must be a singleton")
    base_periods = dataslate.base_periods
    break_points = _get_break_points_in_base_columns(model, dataslate, plan, )
    break_periods = _get_break_periods_from_break_points(break_points, base_periods, )
    break_periods_next = break_periods[1:] + (base_periods[-1] + 1, )
    return tuple(
        Frame(start, next_-1, get_simulation_end(start, next_-1, ), )
        for start, next_ in zip(break_periods, break_periods_next, )
    )


def setup_single_frame(
    model,
    dataslate: Dataslate,
    plan: SimulationPlan | None,
    get_simulation_end: Callable,
) -> tuple[Frame, ...]:
    """
    Return a tuple with a single frame that spans the entire simulation period
    """
    base_periods = dataslate.base_periods
    frame_start = base_periods[0]
    frame_end = base_periods[-1]
    simulation_end = frame_end
    single_frame = Frame(
        frame_start,
        frame_end,
        get_simulation_end(frame_start, frame_end, ),
    )
    return (single_frame, )


def _get_break_points_in_base_columns(
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
    break_points = _np.zeros((len(columns), ), dtype=bool, )
    break_points[0] = True
    if rows and columns:
        cutout = dataslate.get_data_variant(0, )[rows, :][:, columns]
        break_points = _update_break_points(break_points, cutout, )
    #
    if plan is not None:
        cutout = plan.tabulate_registered_points(
            "endogenized_unanticipated", can_be_endogenized, periods,
        )
        break_points = _update_break_points(break_points, cutout, )
        cutout = plan.tabulate_registered_points(
            "exogenized_unanticipated", can_be_exogenized, periods,
        )
        break_points = _update_break_points(break_points, cutout, )
    return break_points


def _get_break_periods_from_break_points(
    break_points: Iterable[bool],
    base_periods: Iterable[Period],
    /,
) -> tuple[Period]:
    after_period = base_periods[-1] + 1
    zipped = zip(base_periods, break_points)
    return tuple(t for t, flag in zipped if flag)


def _update_break_points(break_points, new_array, ):
    """
    """
    if new_array is None:
        return break_points
    new_break_points = _np.any(_np.isfinite(new_array) & (new_array != 0), axis=0, )
    return break_points | new_break_points
