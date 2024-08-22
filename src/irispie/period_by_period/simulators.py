"""
Dynamic nonlinear period-by-period simulator
"""


#[
from __future__ import annotations

import numpy as _np
import scipy as _sp
import functools as _ft
import wlogging as _wl

from ..simultaneous import main as _simultaneous
from ..plans.simulation_plans import (SimulationPlan, )
from ..dataslates.main import (Dataslate, )
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
_METHOD_NAME = "period_by_period"


def simulate(
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

    root_settings = _DEFAULT_SCIPY_ROOT_SETTINGS | (root_settings or {})

    qid_to_name = simulatable_v.create_qid_to_name()
    name_to_qid = simulatable_v.create_name_to_qid()
    all_quantities = simulatable_v.get_quantities()
    #
    wrt_qids = _get_wrt_qids(simulatable_v, name_to_qid, )
    wrt_equations = simulatable_v.get_dynamic_equations(kind=_equations.ENDOGENOUS_EQUATION, )

    if len(wrt_equations) != len(wrt_qids):
        raise _wrongdoings.IrisPieError(
            f"Number of endogenous equations {len(wrt_equations)} "
            f"does not match number of endogenous quantities {len(wrt_qids)}"
        )

    if not wrt_equations:
        return

    terminators = _terminators.create_terminators(simulatable_v, terminal, )

    create_evaluator = _ft.partial(
        _evaluators.create_evaluator,
        wrt_equations=wrt_equations,
        all_quantities=all_quantities,
        terminal_simulator=terminators.terminate_simulation,
        context=simulatable_v.get_context(),
        iter_printer_settings=iter_printer_settings,
    )

    create_evaluator = _ft.lru_cache(maxsize=None, )(create_evaluator, )

    when_missing_stream = _wrongdoings.create_stream(
        when_missing,
        f"These values are missing at the start of simulation: ",
    )

    starter = _ITER_STARTER[start_iter_from]

    catch_missing = _ft.partial(
        _catch_missing,
        dataslate_v=dataslate_v,
        qid_to_name=qid_to_name,
        fallback_value=fallback_value,
        when_missing_stream=when_missing_stream,
    )
    #
    periods = dataslate_v.periods
    data = dataslate_v.get_data_variant(0, )
    when_fails_stream = _wrongdoings.create_stream(
        when_fails,
        "Simulation failed to successfully complete",
    )
    #
    for t in dataslate_v.base_columns:
        current_wrt_qids, current_evaluator = \
            _setup_current_period(plan, create_evaluator, wrt_qids, periods[t], name_to_qid, )
        current_evaluator.iter_printer.header_message = \
            _create_header_message(vid, t, periods[t], )
        data[current_wrt_qids, t] = starter(data, current_wrt_qids, t, )
        catch_missing(data, t, )
        #
        init_guess = current_evaluator.get_init_guess(data, t, )
        root_final = _sp.optimize.root(
            current_evaluator.evaluate, init_guess,
            args=(data, t, ),
            jac=True,
            **root_settings,
        )
        #
        if not root_final.success:
            _catch_fail(when_fails_stream, root_final, periods[t], )
        current_evaluator.update(root_final.x, data, t, )
        #
        current_evaluator.iter_printer.print_footer()
        current_evaluator.iter_printer.reset()
    #
    when_fails_stream._raise()
    #
    info = {
        "method": _METHOD_NAME,
    }
    #
    return info


def _setup_current_period(
    plan: SimulationPlan | None,
    create_evaluator: Callable,
    current_wrt_qids: tuple[int, ...],
    current_period: Period,
    name_to_qid: dict[str, int],
    /,
) -> tuple[tuple[int, ...], PeriodEvaluator]:
    """
    """
    current_wrt_qids = tuple(current_wrt_qids)
    if plan:
        names_exogenized = plan.get_exogenized_unanticipated_in_period(current_period, )
        names_endogenized = plan.get_endogenized_unanticipated_in_period(current_period, )
        if len(names_exogenized) != len(names_endogenized):
            raise _wrongdoings.IrisPieCritical(
                f"Number of exogenized quantities {len(names_exogenized)}"
                f" does not match number of endogenized quantities {len(names_endogenized)}"
                f" in period {current_period}"
            )
        qids_exogenized = tuple(name_to_qid[name] for name in names_exogenized)
        qids_endogenized = tuple(name_to_qid[name] for name in names_endogenized)
        current_wrt_qids = tuple(sorted(set(current_wrt_qids).difference(qids_exogenized).union(qids_endogenized)))
    current_evaluator = create_evaluator(current_wrt_qids, )
    return current_wrt_qids, current_evaluator


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
    t: int,
    /,
    dataslate_v: Dataslate,
    qid_to_name: dict[int, str],
    fallback_value: Real,
    when_missing_stream: _wrongdoings.Stream,
) -> None:
    """
    """
    missing = _np.isnan(data[:, t])
    if not missing.any():
        return
    #
    data[missing, t] = fallback_value
    #
    current_period = dataslate_v.periods[t]
    shift = 0
    for qid in _np.flatnonzero(missing):
        when_missing_stream.add(
            f"{qid_to_name[qid]}[{current_period+shift}]"
            f" when simulating {current_period}"
        )


def _catch_fail(
    when_fails_stream,
    root_final: _sp.optimize.OptimizeResult,
    period: Period,
) -> None:
    """
    """
    when_fails_stream.add(
        f"Simulation failed in {period} with message: {root_final.message}"
    )


def _create_header_message(
    vid: str,
    t: int,
    current_period: int,
) -> str:
    """
    """
    return f"[Variant {vid}][Period {current_period}]"


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


_DEFAULT_SCIPY_ROOT_SETTINGS = {
    "method": "hybr",
    "tol": 1e-12,
    "options": {"xtol": 1e-12, },
}

