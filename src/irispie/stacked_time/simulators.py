"""
Dynamic nonlinear stacked-time simulator
"""


#[
from __future__ import annotations

import warnings as _wa
import numpy as _np
import scipy as _sp
import functools as _ft
import itertools as _it
import wlogging as _wl

from ..simultaneous import main as _simultaneous
from ..plans.simulation_plans import (SimulationPlan, )
from ..dataslates.main import (Dataslate, )
from ..incidences import main as _incidences
from ..incidences.main import (Token, )
from .. import equations as _equations
from .. import equations as _equations
from .. import wrongdoings as _wrongdoings
from ..fords.terminators import (Terminator, )

from . import _evaluators as _evaluators

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from numbers import (Real, )
    from typing import (Callable, Any, Literal, )
    from ..dates import (Period, )
#]


_DEFAULT_FALLBACK_VALUE = 1/9
_METHOD_NAME = "stacked_time"


def _simulate_frame(
    simulatable_v: _simultaneous.Simultaneous,
    dataslate_v: Dataslate,
    plan: SimulationPlan | None,
    vid: int,
    *,
    logger: _wl.Logger,
    return_info: bool = False,
    #
    start_iter_from: Literal["data", "first_order", "previous_period"] = "previous_period",
    root_settings: dict[str, Any] | None = None,
    iter_printer_settings: dict[str, Any] | None = None,
    when_fails: Literal["critical", "error", "warning", "silent"] = "critical",
    when_missing: Literal["critical", "error", "warning", "silent"] = "critical",
    fallback_value: Real = _DEFAULT_FALLBACK_VALUE,
    terminal: Literal["data", "first_order", ] = "first_order",
) -> dict[str, Any]:
    """
    """

    if not simulatable_v.is_singleton:
        raise ValueError("Simulator requires a singleton simulatable object")
    #
    root_settings = (
        _DEFAULT_ROOT_SETTINGS
        if root_settings is None
        else _DEFAULT_ROOT_SETTINGS | root_settings
    )
    #
    when_missing_stream = \
        _wrongdoings.STREAM_FACTORY[when_missing] \
        (f"These values are missing at the start of simulation: ")

    max_lead = simulatable_v.max_lead
    qid_to_name = simulatable_v.create_qid_to_name()
    name_to_qid = simulatable_v.create_name_to_qid()
    all_quantities = simulatable_v.get_quantities()


    endogenous_qids = _get_sorted_endogenous_qids(simulatable_v, name_to_qid, )
    wrt_equations = simulatable_v.get_dynamic_equations(kind=_equations.ENDOGENOUS_EQUATION, )


    periods = dataslate_v.periods
    columns_to_run = tuple(dataslate_v.base_columns)

    if len(wrt_equations) != len(endogenous_qids):
        raise _wrongdoings.IrisPieCritical(
            f"Number of endogenous equations {len(wrt_equations)} "
            f"does not match number of endogenous quantities {len(endogenous_qids)}"
        )

    if not wrt_equations:
        return

    needs_terminal = terminal == "first_order" and max_lead

    iter_printer_settings = {"every": 1, }
    # create_evaluator_closure = _ft.lru_cache(maxsize=None, )(create_evaluator_closure, )

    # starter = _ITER_STARTER[start_iter_from]

    catch_missing = _ft.partial(
        _catch_missing,
        qid_to_name=qid_to_name,
        fallback_value=fallback_value,
        when_missing_stream=when_missing_stream,
        periods=periods,
    )

    data = dataslate_v.get_data_variant(0, )

    # Create list of periods for reporting purposes
    current_periods = tuple(periods[i] for i in columns_to_run)

    current_wrt_spots = _get_current_wrt_spots(
        plan=plan,
        endogenous_qids=endogenous_qids,
        columns_to_run=columns_to_run,
        current_periods=current_periods,
        name_to_qid=name_to_qid,
    )

    terminator = None
    if needs_terminal:
        terminator = Terminator(simulatable_v, columns_to_run, wrt_equations, )
        terminator.create_terminal_jacobian_map(current_wrt_spots, )

    current_evaluator = _evaluators.create_evaluator_closure(
        current_wrt_spots,
        columns_to_run,
        wrt_equations=wrt_equations,
        all_quantities=all_quantities,
        terminator=terminator,
        context=simulatable_v.get_context(),
        #iter_printer_settings=iter_printer_settings,
        iter_printer_settings={"every": 1, },
    )



    # data[current_wrt_qids, t] = starter(data, current_wrt_qids, t, )
    # ford_extender(data, columns_to_run[0]-1, num_columns_to_run, )

    catch_missing(data, columns_to_run, )

    header_message = _create_header_message(vid, current_periods, )
    # print(header_message, )

    data0 = data.copy()
    # guess = current_evaluator.get_init_guess(data, )
    # for i in range(7):
    #     f, j = current_evaluator.eval_func_jacob(guess, data, )
    #     unit_step = - _sp.sparse.linalg.spsolve(j, f)
    #     opt_step, min_f = None, None
    #     ff = []
    #     #for j in (0.1, 0.5, 1.0, 1.2, 1.5, 1.8):
    #     #    candidate = guess + j*unit_step
    #     #    f, _ = current_evaluator.eval_func_jacob(candidate, data, columns_to_run, )
    #     #    max_f = _np.abs(f).max()
    #     #    ff.append(max_f)
    #     #    if min_f is None or max_f < min_f:
    #     #        min_f = max_f
    #     #        opt_step = j
    #     opt_step = 1
    #     guess = guess + opt_step*unit_step

    info = {
        "method": _METHOD_NAME,
        "evaluator": current_evaluator,
        "data": data0,
    }

    return info


def _get_current_wrt_spots(
    plan: SimulationPlan | None,
    endogenous_qids: Iterable[int],
    columns_to_run: Iterable[int],
    name_to_qid: dict[str, int],
    current_periods: Period,
) -> tuple[int, ...]:
    """
    """
    current_wrt_spots = tuple(
        Token(qid, column, )
        for column, qid in _it.product(columns_to_run, endogenous_qids, )
    )
    if plan is None:
        return current_wrt_spots
    names_exogenized = plan.get_exogenized_unanticipated_in_period(current_periods, )
    names_endogenized = plan.get_endogenized_unanticipated_in_period(current_periods, )
    if len(names_exogenized) != len(names_endogenized):
        raise _wrongdoings.IrisPieCritical(
            f"Number of exogenized quantities {len(names_exogenized)}"
           f" does not match number of endogenized quantities {len(names_endogenized)}"
            f" in periods {current_periods[0]}>>{current_periods[-1]}"
        )
    qids_exogenized = tuple(name_to_qid[name] for name in names_exogenized)
    qids_endogenized = tuple(name_to_qid[name] for name in names_endogenized)
    current_wrt_spots = tuple(sorted(set(current_wrt_spots).difference(qids_exogenized).union(qids_endogenized)))
    return current_wrt_spots


def _start_iter_from_previous_period(
    data: _np.ndarray,
    wrt_qids: tuple[int, ...],
    t: int,
) -> _np.ndarray:
    """
    """
    source_t = t - 1 if t > 0 else 0
    return data[wrt_qids, source_t]


def _start_iter_from_data(
    data: _np.ndarray,
    wrt_qids: tuple[int, ...],
    t: int,
) -> _np.ndarray:
    """
    """
    return data[wrt_qids, t]


def _start_iter_from_first_order(
    data: _np.ndarray,
    wrt_qid: tuple[int, ...],
    t: int,
) -> _np.ndarray:
    """
    """
    raise NotImplementedError()


def _catch_missing(
    data: _np.ndarray,
    columns: int,
    /,
    qid_to_name: dict[int, str],
    when_missing_stream: _wrongdoings.Stream,
    periods: Iterable[Period],
    fallback_value: Real,
) -> None:
    """
    """
    missing = _np.isnan(data[:, columns])
    if not missing.any():
        return
    missing_qids, *_ = missing.any(axis=1, ).nonzero()
    start = periods[columns[0]]
    end = periods[columns[-1]]
    for qid in missing_qids:
        message = f"{qid_to_name[qid]} when simulating {start}>>{end}"
        when_missing_stream.add(message, )
    data[:, columns][missing] = fallback_value


def _catch_fail(
    root_final: _sp.optimize.OptimizeResult,
    when_fails: Literal["error", "warning", "silent"],
) -> None:
    """
    """
    raise ValueError()


def _create_header_message(
    vid: str,
    current_periods: int,
) -> str:
    """
    """
    start = current_periods[0]
    end = current_periods[-1]
    return f"[Variant {vid}][Periods {start}>>{end}]"


def _get_sorted_endogenous_qids(
    simulatable_v,
    name_to_qid: dict[str, int],
    /,
) -> tuple[int, ...]:
    """
    """
    plannable = simulatable_v.get_simulation_plannable()
    wrt_names = set(
        tuple(plannable.can_be_exogenized_unanticipated)
        + tuple(plannable.can_be_exogenized_anticipated)
    )
    return sorted(name_to_qid[name] for name in wrt_names)


_ITER_STARTER = {
    "previous_period": _start_iter_from_previous_period,
    "data": _start_iter_from_data,
    "first_order": _start_iter_from_first_order,
}


_DEFAULT_ROOT_SETTINGS = {
    "method": "hybr",
    "tol": 1e-12,
    # "options": {"col_deriv": True},
}


simulate = _simulate_frame

