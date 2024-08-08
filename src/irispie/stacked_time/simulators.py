"""
Dynamic nonlinear period-by-period simulator
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
from ..incidences.main import (Token, )
from .. import equations as _equations
from .. import equations as _equations
from .. import wrongdoings as _wrongdoings
from ..fords import terminators as _terminators

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
    start_iter_from: Literal["data", "first_order", "previous_period"] = "previous_period",
    root_settings: dict[str, Any] | None = None,
    iter_printer_settings: dict[str, Any] | None = None,
    when_fails: Literal["critical", "error", "warning", "silent"] = "critical",
    when_missing: Literal["critical", "error", "warning", "silent"] = "critical",
    fallback_value: Real = _DEFAULT_FALLBACK_VALUE,
    terminal: Literal["data", "first_order", ] = "first_order",
    return_info: bool = False,
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

    max_lead = simulatable_v.max_lead
    qid_to_name = simulatable_v.create_qid_to_name()
    name_to_qid = simulatable_v.create_name_to_qid()
    all_quantities = simulatable_v.get_quantities()
    #
    wrt_qids = _get_wrt_qids(simulatable_v, name_to_qid, )
    wrt_equations = simulatable_v.get_dynamic_equations(kind=_equations.ENDOGENOUS_EQUATION, )

    periods = dataslate_v.periods
    columns_to_run = _np.array(dataslate_v.base_columns, dtype=int, )
    num_columns_to_run = len(columns_to_run)
    terminal_columns = tuple(range(
        columns_to_run[-1] + 1,
        columns_to_run[-1] + max_lead,
    ))

    if len(wrt_equations) != len(wrt_qids):
        raise _wrongdoings.IrisPieCritical(
            f"Number of endogenous equations {len(wrt_equations)} "
            f"does not match number of endogenous quantities {len(wrt_qids)}"
        )

    if not wrt_equations:
        return

    terminators = (
        _terminators.create_terminators(simulatable_v, )
        if terminal == "first_order" else None
    )

    # _wa.warn("!!! Terminal simulator deactivated")
    # terminate_simulation = None

    create_evaluator_closure = _ft.partial(
        _evaluators.create_evaluator_closure,
        wrt_equations=wrt_equations,
        all_quantities=all_quantities,
        terminate_simulation=terminate_simulation,
        context=simulatable_v.get_context(),
        iter_printer_settings=iter_printer_settings,
    )

    create_evaluator_closure = _ft.lru_cache(maxsize=None, )(create_evaluator_closure, )

    when_missing_stream = \
        _wrongdoings.STREAM_FACTORY[when_missing] \
        (f"These values are missing at the start of simulation: ")

    # starter = _ITER_STARTER[start_iter_from]

    catch_missing = _ft.partial(
        _catch_missing,
        qid_to_name=qid_to_name,
        fallback_value=fallback_value,
        when_missing_stream=when_missing_stream,
        periods=periods,
    )

    data = dataslate_v.get_data_variant(0, )

    current_periods = tuple(periods[i] for i in columns_to_run)
    first_column = columns_to_run[0]
    wrt_tokens = tuple(
        Token(qid, column-first_column, )
        for column, qid in _it.product(columns_to_run, wrt_qids, )
    )
    terminal_wrt_tokens = tuple(
        Token(qid, column-first_column, )
        for column, qid in _it.product(terminal_columns, wrt_qids, )
    )

    current_wrt_qids, current_evaluator = _setup_current_period(
        plan,
        create_evaluator_closure,
        wrt_tokens,
        terminal_wrt_tokens,
        current_periods,
        name_to_qid,
        num_columns_to_run,
    )

    a, b = current_evaluator.evaluate(None, data, columns_to_run, )

    # data[current_wrt_qids, t] = starter(data, current_wrt_qids, t, )
    # ford_extender(data, columns_to_run[0]-1, num_columns_to_run, )

    catch_missing(data, columns_to_run, )
    #
    data0 = data.copy()
    # init_guess = current_evaluator.get_init_guess(data, columns_to_run, )

    # root_final = _sp.optimize.root(
    #     current_evaluator.evaluate, init_guess,
    #     args=(data, columns_to_run, ),
    #     jac=True,
    #     **root_settings,
    # )

    header_message = _create_header_message(vid, current_periods, )
    print(header_message, )

    print("*"*80)
    data = data0
    guess = current_evaluator.get_init_guess(data, columns_to_run, )
    for i in range(10):
        f, j = current_evaluator.evaluate(guess, data, columns_to_run, )
        # import IPython; IPython.embed()
        #j0 = j[0,0:42]
        #index = j0.nonzero()
        #print(j0[*index])
        unit_step = - _np.linalg.solve(j, f)
        opt_step, min_f = None, None
        ff = []
        #for j in (0.1, 0.5, 1.0, 1.2, 1.5, 1.8):
        #    candidate = guess + j*unit_step
        #    f, _ = current_evaluator.evaluate(candidate, data, columns_to_run, )
        #    max_f = _np.abs(f).max()
        #    ff.append(max_f)
        #    if min_f is None or max_f < min_f:
        #        min_f = max_f
        #        opt_step = j
        opt_step = 1
        guess = guess + opt_step*unit_step




    #
    # if not root_final.success:
        # _catch_fail(root_final, when_fails, )
    # current_evaluator.update(root_final.x, data, t, )
    # #
    # current_evaluator.iter_printer.print_footer()
    # current_evaluator.iter_printer.reset()
    #
    info = {
        "method": _METHOD_NAME,
    }
    #
    return info


def _setup_current_period(
    plan: SimulationPlan | None,
    create_evaluator_closure: Callable,
    current_wrt_tokens: tuple[int, ...],
    terminal_wrt_tokens: tuple[int, ...],
    current_periods: Period,
    name_to_qid: dict[str, int],
    num_columns_to_run: int,
    /,
) -> tuple[tuple[int, ...], ...]:
    """
    """
    current_wrt_tokens = tuple(current_wrt_tokens)
    if plan:
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
        current_wrt_tokens = tuple(sorted(set(current_wrt_tokens).difference(qids_exogenized).union(qids_endogenized)))
    current_evaluator = create_evaluator_closure(current_wrt_tokens, terminal_wrt_tokens, num_columns_to_run, )
    return current_wrt_tokens, current_evaluator


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
    data[:, columns][missing] = fallback_value
    missing_qids, *_ = missing.any(axis=1, ).nonzero()
    start = periods[columns[0]]
    end = periods[columns[-1]]
    for qid in missing_qids:
        message = f"{qid_to_name[qid]} when simulating {start}>>{end}"
        when_missing_stream.add(message, )


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


def _get_wrt_qids(
    simulatable_v,
    name_to_qid: dict[str, int],
    /,
) -> tuple[int, ...]:
    """
    """
    plannable = simulatable_v.get_simulation_plannable()
    wrt_names \
        = tuple(plannable.can_be_exogenized_unanticipated) \
        + tuple(plannable.can_be_exogenized_anticipated)
    wrt_qids = sorted(
        name_to_qid[name]
        for name in set(wrt_names)
    )
    return wrt_qids


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

