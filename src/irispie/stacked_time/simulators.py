"""
Dynamic nonlinear stacked-time simulator
"""


#[

from __future__ import annotations

import warnings as _wa
import numpy as _np
import scipy as _sp
import itertools as _it
import neqs as _nq

from ..simultaneous import main as _simultaneous
from ..plans.simulation_plans import SimulationPlan
from ..dataslates.main import Dataslate
from ..incidences import main as _incidences
from ..incidences.main import Token
from .. import quantities as _quantities
from .. import equations as _equations
from .. import frames as _frames
from .. import dates as _dates
from ..dates import Span
from ..frames import SingleFrame
from .. import wrongdoings as _wrongdoings
from ..fords import simulators as _ford_simulators
from ..fords.terminators import Terminator

from . import _evaluators as _evaluators
from . import _iter_printers as _iter_printers

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Real
    from typing import Callable, Any, Literal
    from ..dates import Period
    from ..simultaneous.main import Simultaneous
    from ..frames import Frame

#]


# Simulator protocol requirements
METHOD_NAME = "stacked_time"
def create_frames(): ...
def simulate_initial_guess(): ...
def simulate_frame(): ...


_DEFAULT_FALLBACK_VALUE = 1/9


_RELEVANT_REGISTER_NAMES = (
    "exogenized_anticipated",
    "endogenized_anticipated",
    "exogenized_unanticipated",
    "endogenized_unanticipated",
)


def create_frames(
    model_v: Simultaneous,
    dataslate_v: Dataslate,
    plan: SimulationPlan | None,
    **kwargs,
) -> tuple[Frame, ...]:
    """
    """
    #[
    base_end = dataslate_v.base_periods[-1]
    return _frames.split_into_frames(
        model_v, dataslate_v, plan,
        get_simulation_end=lambda *_, : base_end,
    )
    #]


def simulate_initial_guess(
    model_v: Simultaneous,
    dataslate_v: Dataslate,
    plan: SimulationPlan | None,
    *,
    initial_guess: Literal["first_order", "data"] = "first_order",
    **kwargs,
) -> None:
    """
    """
    _INITIAL_GUESS_SIMULATOR[initial_guess](model_v, dataslate_v, )


def simulate_frame(
    model_v: Simultaneous,
    frame_ds: Dataslate,
    *,
    frame: Frame,
    input_data_array: _np.ndarray,
    plan: SimulationPlan | None,
    simulation_header: str,
    return_info: bool = False,
    # Method specific settings
    solver_settings: dict[str, Any] | None = None,
    when_fails: Literal["critical", "error", "warning", "silent"] = "error",
    when_missing: Literal["critical", "error", "warning", "silent"] = "error",
    fallback_value: Real = _DEFAULT_FALLBACK_VALUE,
    terminal: Literal["data", "first_order", ] = "first_order",
    _precatch_missing: Callable | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    """
    #[

    solver_settings = solver_settings or {}
    solver_settings = {"norm_order": float("inf"), } | solver_settings

    when_missing_stream = \
        _wrongdoings.STREAM_FACTORY[when_missing] \
        (f"These values are missing at the start of the {frame} simulation: ")

    all_quantities = model_v.get_quantities()
    qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
    qid_to_name = model_v.create_qid_to_name()
    name_to_qid = model_v.create_name_to_qid()

    endogenous_quantities = model_v.get_quantities(kind=_quantities.ENDOGENOUS_VARIABLE, )
    endogenous_qids = tuple(i.id for i in endogenous_quantities)
    wrt_equations = model_v.get_dynamic_equations(kind=_equations.ENDOGENOUS_EQUATION, )

    if len(wrt_equations) != len(endogenous_qids):
        raise _wrongdoings.IrisPieCritical(
            f"Number of endogenous equations {len(wrt_equations)} "
            f"does not match number of endogenous quantities {len(endogenous_qids)}"
        )

    if not wrt_equations:
        success = True
        return success

    periods = frame_ds.periods
    columns_to_run = tuple(range(frame.first, frame.simulation_last+1, ))
    periods_to_run = tuple(periods[i] for i in columns_to_run)

    wrt_spots, exogenized_spots = _get_wrt_spots(
        plan=plan,
        endogenous_qids=endogenous_qids,
        columns_to_run=columns_to_run,
        periods_to_run=periods_to_run,
        name_to_qid=name_to_qid,
    )

    terminator = None
    needs_terminator = terminal == "first_order" and model_v.max_lead
    if needs_terminator:
        terminator = Terminator(model_v, columns_to_run, wrt_equations, )
        terminator.create_terminal_jacobian_map(wrt_spots, )

    evaluator = _evaluators.create_evaluator(
        wrt_spots=wrt_spots,
        columns_to_eval=columns_to_run,
        wrt_equations=wrt_equations,
        all_quantities=all_quantities,
        terminator=terminator,
        context=model_v.get_context(),
    )

    data = frame_ds.get_data_variant()

    if _precatch_missing is not None:
        _precatch_missing(data, wrt_spots, )

    _catch_missing(
        data=data,
        wrt_spots=wrt_spots,
        frame=frame,
        qid_to_name=qid_to_name,
        fallback_value=fallback_value,
        when_missing_stream=when_missing_stream,
        periods=periods,
    )

    _copy_exogenized_data_to_frame_data(data, exogenized_spots, input_data_array, )

    iter_printer = _iter_printers.create_iter_printer(
        equations=wrt_equations,
        qids=tuple(tok.qid for tok in wrt_spots),
        qid_to_logly=qid_to_logly,
        qid_to_name=qid_to_name,
        custom_header=simulation_header,
        **solver_settings,
    )

    init_guess = evaluator.get_init_guess(data, )

    final_guess, exit_status = _nq.damped_newton(
        eval_func=evaluator.eval_func,
        eval_jacob=evaluator.eval_jacob,
        init_guess=init_guess,
        iter_printer=iter_printer,
        args=(data, ),
        **solver_settings,
    )

    evaluator.update(final_guess, data, )
    return exit_status

    #]


def _get_wrt_spots(
    plan: SimulationPlan | None,
    endogenous_qids: Iterable[int],
    columns_to_run: Iterable[int],
    periods_to_run: Iterable[Period],
    name_to_qid: dict[str, int],
) -> tuple[tuple[Token, ...], set[Token, ...]]:
    """
    """
    wrt_spots = tuple(
        Token(qid, column, )
        for column, qid in _it.product(columns_to_run, endogenous_qids, )
    )
    if plan is None:
        return wrt_spots, None,
    #
    registers_as_bool_arrays = plan.get_registers_as_bool_arrays(
        periods=periods_to_run,
        register_names=_RELEVANT_REGISTER_NAMES,
    )
    row_names_in_registers = {
        n: tuple(plan.get_register_by_name(n, ).keys())
        for n in _RELEVANT_REGISTER_NAMES
    }
    base_first = columns_to_run[0]
    def spots_from_register(register_name: str, columns_to_run: Iterable[int], ) -> set[Token]:
        return set(
            Token(name_to_qid[n], shift, )
            for row, n in enumerate(row_names_in_registers[register_name])
            for column, shift in enumerate(columns_to_run)
            if registers_as_bool_arrays[register_name][row, column]
        )
    exogenized_spots = (
        spots_from_register("exogenized_anticipated", columns_to_run, )
        | spots_from_register("exogenized_unanticipated", columns_to_run[0:1], )
    )
    endogenized_spots = (
        spots_from_register("endogenized_anticipated", columns_to_run, )
        | spots_from_register("endogenized_unanticipated", columns_to_run[0:1], )
    )
    wrt_spots = tuple(sorted(
        set(wrt_spots)
        .difference(exogenized_spots)
        .union(endogenized_spots)
    ))
    return wrt_spots, exogenized_spots


def _catch_missing(
    data: _np.ndarray,
    wrt_spots: Iterable[Token],
    frame: Frame,
    qid_to_name: dict[int, str],
    when_missing_stream: _wrongdoings.Stream,
    periods: Iterable[Period],
    fallback_value: Real,
) -> None:
    """
    """
    #[
    missing = _np.isnan(data)
    if not missing.any():
        return
    for qid, column in wrt_spots:
        if not missing[qid, column]:
            continue
        when_missing_stream.add(f"{qid_to_name[qid]}[{periods[column]}]", )
        data[qid, column] = fallback_value
    when_missing_stream._raise()
    #]


def _catch_fail(
    root_final: _sp.optimize.OptimizeResult,
    when_fails: Literal["error", "warning", "silent"],
) -> None:
    """
    """
    raise ValueError()


def _simulate_initial_guess_first_order(
    model_v: Simultaneous,
    frame_ds: DataSlate,
) -> None:
    """
    Simulate initial guess by running a first-order simulation without any
    shocks or conditioning
    """
    frame = SingleFrame(frame_ds.base_periods[0], frame_ds.base_periods[-1], )
    frame.resolve_columns(frame_ds.periods[0], )
    _ford_simulators.simulate_flat(
        model_v, frame_ds, frame,
        deviation=False,
        ignore_shocks=True,
    )


def _simulate_initial_guess_data(
    *args, **kwargs,
) -> None:
    """
    Initial guess is the data
    """
    pass


_INITIAL_GUESS_SIMULATOR = {
    "first_order": _simulate_initial_guess_first_order,
    "data": _simulate_initial_guess_data,
}


def _copy_exogenized_data_to_frame_data(
    data: _np.ndarray,
    exogenized_spots: set[Token],
    input_data_array: _np.ndarray,
) -> None:
    """
    """
    if not exogenized_spots:
        return
    indexes = tuple(zip(*exogenized_spots))
    data[indexes] = input_data_array[indexes]

