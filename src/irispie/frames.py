"""
Time frames for dynamic simulations
"""


#[

from __future__ import annotations

import numpy as _np

from .dates import SPAN_ELLIPSIS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import (Callable, Iterable, )
    from .dataslates.main import Dataslate
    from .plans.simulation_plans import SimulationPlan
    from .dates import Period

#]


class Frame:
    """
    """
    #[

    __slots__ = (
        # Periods
        "start",
        "end",
        "simulation_end",
        # Columns
        "first",
        "last",
        "simulation_last",
        "slice",
        "zero_unanticipated_slice",
        "simulation_slice",
        "num_simulation_columns",
    )

    def __init__(
        self,
        start: Period,
        end: Period,
        simulation_end: Period | None = None,
    ) -> None:
        """
        """
        self.start = start
        self.end = end
        self.simulation_end = simulation_end
        self.first = None
        self.last = None
        self.simulation_last = None
        self.slice = None
        self.simulation_slice = None
        self.num_simulation_columns = None

    def resolve_columns(
        self,
        first_column_period: Period,
    ) -> None:
        """
        Given the time period of the first column of data array, determine the
        first and last columns of the frame
        """
        self.first = self.start - first_column_period
        self.last = self.end - first_column_period
        self.simulation_last = self.simulation_end - first_column_period
        self.slice = slice(self.first, self.last+1, )
        self.zero_unanticipated_slice = slice(self.first+1, None, )
        self.simulation_slice = slice(self.first, self.simulation_last+1, )
        self.num_simulation_columns = self.simulation_last - self.first + 1


class SplitFrame(Frame, ):
    """
    """
    #[

    def __repr__(self, ) -> str:
        """
        """
        return f"<SplitFrame {self.start}{SPAN_ELLIPSIS}{self.end}{SPAN_ELLIPSIS}{self.simulation_end}>"

    def prune_frame_data(
        self,
        frame_ds: Dataslate,
        unanticipated_qids: Iterable[int],
    ) -> None:
        """
        Remove unanticipated shocks after the first period of the frame
        """
        if self.start == self.simulation_end:
            return
        data = frame_ds.get_data_variant()
        data[unanticipated_qids, self.zero_unanticipated_slice] = 0

    def write_frame_data_to_main_dataslate(
        self,
        main_ds: Dataslate,
        frame_ds: Dataslate,
        unanticipated_shock_qids: Iterable[int],
    ) -> None:
        """
        Copy frame columns from frame_ds to main_ds
        """
        main_data = main_ds.get_data_variant(0, )
        frame_data = frame_ds.get_data_variant(0, )
        all_qids = range(frame_ds.num_names)
        regular_qids = tuple(
            qid for qid in all_qids
            if qid not in unanticipated_shock_qids
        )
        main_data[regular_qids, self.slice] = frame_data[regular_qids, self.slice]
        main_data[unanticipated_shock_qids, self.first] = frame_data[unanticipated_shock_qids, self.first]

    #]


class SingleFrame(Frame, ):
    """
    Single frame over the entire simulation span with all unanticipated data retained
    """
    #[

    def __init__(self, start: Period, end: Period, ) -> None:
        """
        """
        super().__init__(start, end, end, )

    def __repr__(self, ) -> str:
        """
        """
        return f"<SingleFrame {self.start}…{self.end}>"

    def prune_frame_data(
        self,
        frame_ds: Dataslate,
        model: Slatable,
    ) -> None:
        r"""
        No pruning, keep unanticipated shocks
        """
        pass

    def write_frame_data_to_main_dataslate(
        self,
        main_ds: Dataslate,
        frame_ds: Dataslate,
        unanticipated_shock_qids: Iterable[int],
    ) -> None:
        """
        """
        main_data = main_ds.get_data_variant(0, )
        frame_data = frame_ds.get_data_variant(0, )
        main_data[:, self.slice] = frame_data[:, self.slice]

    #]


def split_into_frames(
    model,
    dataslate: Dataslate,
    plan: SimulationPlan | None,
    get_simulation_end: Callable,
) -> tuple[bool, ...]:
    """
    """
    #[
    base_break_points = _populate_base_break_points(model, dataslate, plan, )
    return split_into_frames_by_breakpoints(
        base_break_points, dataslate, get_simulation_end,
    )
    #]


def setup_single_frame(
    model,
    dataslate: Dataslate,
    plan: SimulationPlan | None,
    **kwargs,
) -> tuple[SingleFrame]:
    """
    Return a tuple with a SingleFrame that spans the entire simulation period
    """
    #[
    base_periods = dataslate.base_periods
    first_dataslate_period = dataslate.periods[0]
    frame_start = base_periods[0]
    frame_end = base_periods[-1]
    simulation_end = frame_end
    frame = SingleFrame(frame_start, frame_end, )
    frame.resolve_columns(first_dataslate_period, )
    return (frame, )
    #]


def split_into_frames_by_breakpoints(
    base_break_points: tuple[bool, ...],
    dataslate: Dataslate,
    get_simulation_end: Callable,
) -> tuple[bool, ...]:
    """
    """
    #[
    base_periods = dataslate.base_periods
    break_periods = _get_break_periods_from_break_points(base_break_points, base_periods, )
    break_periods_next = break_periods[1:] + (base_periods[-1] + 1, )
    first_dataslate_period = dataslate.periods[0]
    frames = []
    for start, next_ in zip(break_periods, break_periods_next, ):
        end = next_ - 1
        new_frame = SplitFrame(start, end, get_simulation_end(start, end, ), )
        new_frame.resolve_columns(first_dataslate_period, )
        frames.append(new_frame, )
    return tuple(frames, )
    #]


def create_empty_base_break_points(dataslate: Dataslate, /, ) -> _np.ndarray:
    """
    """
    #[
    base_columns = dataslate.base_columns
    return _np.zeros((len(base_columns), ), dtype=bool, )
    #]


def _populate_base_break_points(
    model,
    dataslate: Dataslate | None,
    plan: SimulationPlan | None,
) -> tuple[Frame]:
    """
    """
    #[
    if not model.is_singleton:
        raise ValueError("Model must be a singleton")
    if not dataslate.is_singleton:
        raise ValueError("Dataslate must be a singleton")
    #
    base_break_points = create_empty_base_break_points(dataslate, )
    base_break_points[0] = True
    plannable = model.get_simulation_plannable()
    #
    # First, get breakpoints at the dates of unanticipated shocks
    name_to_row = dataslate.create_name_to_row()
    rows = tuple(
        name_to_row[name]
        for name in plannable.can_be_endogenized_unanticipated
    )
    base_columns = dataslate.base_columns
    if rows and base_columns:
        cutout = dataslate.get_data_variant(0, )[rows, :][:, base_columns]
        base_break_points = _update_break_points(base_break_points, cutout, )
    #
    # Then, get breakpoints at the dates of endogenized unanticipated shocks
    if plan is not None:
        base_periods = dataslate.base_periods
        cutout = plan.get_register_as_bool_array(
            "endogenized_unanticipated",
            plannable.can_be_endogenized_unanticipated,
            base_periods,
        )
        base_break_points = _update_break_points(base_break_points, cutout, )
    #
    return base_break_points
    #]


def _get_break_periods_from_break_points(
    base_break_points: Iterable[bool],
    base_periods: Iterable[Period],
    /,
) -> tuple[Period]:
    """
    """
    #[
    after_period = base_periods[-1] + 1
    zipped = zip(base_periods, base_break_points)
    return tuple(t for t, flag in zipped if flag)
    #]


def _update_break_points(base_break_points, new_array, ):
    """
    """
    #[
    if new_array is None:
        return base_break_points
    new_break_points = _np.any(_np.isfinite(new_array) & (new_array != 0), axis=0, )
    return base_break_points | new_break_points
    #]

