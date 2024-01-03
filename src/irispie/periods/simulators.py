"""
Dynamic nonlinear period-by-period simulator
"""


#[
from __future__ import annotations

from numbers import (Real, )
from typing import (Callable, Any, Literal, )
import numpy as _np
import scipy as _sp
import functools as _ft

from ..evaluators.base import (DEFAULT_INIT_GUESS, )
from ..simultaneous import main as _simultaneous
from ..plans import main as _plans
from ..dataslates import main as _dataslates
from .. import quantities as _quantities
from .. import equations as _equations
from .. import wrongdoings as _wrongdoings
from . import evaluators as _evaluators
#]


_DEFAULT_FALLBACK_VALUE = 1/9


def simulate(
    model: _simultaneous.Simultaneous,
    dataslate: _dataslate.Dataslate,
    vid: int,
    /,
    *,
    plan: _plans.PlanSimulate | None,
    start_iter_from: Literal["data", "first_order", "previous_period"] = "previous_period",
    root_settings: dict[str, Any] | None = None,
    iter_printer_settings: dict[str, Any] | None = None,
    when_fails: Literal["critical", "error", "warning", "silent"] = "critical",
    when_missing: Literal["critical", "error", "warning", "silent"] = "critical",
    fallback_value: Real = _DEFAULT_FALLBACK_VALUE,
) -> None:
    """
    """
    root_settings = (
        _DEFAULT_ROOT_SETTINGS
        if root_settings is None
        else _DEFAULT_ROOT_SETTINGS | root_settings
    )

    qid_to_name = model.create_qid_to_name()
    name_to_qid = model.create_name_to_qid()
    all_quantities = model.get_quantities()
    endogenous_names = model.simulate_can_be_exogenized
    exogenous_names = model.simulate_can_be_endogenized
    endogenous_qids = tuple(name_to_qid[name] for name in endogenous_names)
    exogenous_qids = tuple(name_to_qid[name] for name in exogenous_names)
    wrt_qids = endogenous_qids
    wrt_equations = model.get_dynamic_equations(kind=_equations.EquationKind.ENDOGENOUS_EQUATION, )

    if len(wrt_equations) != len(wrt_qids):
        raise _wrongdoings.IrisPieError(
            f"Number of endogenous equations {len(wrt_equations)} "
            f"does not match number of endogenous quantities {len(wrt_qids)}"
        )

    if len(wrt_equations) == 0:
        return

    evaluator_factory = _ft.partial(
        _evaluators.PeriodEvaluator,
        wrt_equations=wrt_equations,
        all_quantities=all_quantities,
        context=model.get_context(),
        iter_printer_settings=iter_printer_settings,
    )

    base_evaluator = evaluator_factory(wrt_qids, )

    when_missing_stream = \
        _wrongdoings.STREAM_FACTORY[when_missing] \
        (f"These values are missing at the start of simulation: ")

    starter = _ITER_STARTER[start_iter_from]

    catch_missing = _ft.partial(
        _catch_missing,
        dataslate=dataslate,
        qid_to_name=qid_to_name,
        fallback_value=fallback_value,
        when_missing_stream=when_missing_stream,
    )

    for t in dataslate.base_columns:
        current_wrt_qids, current_evaluator = \
            _set_up_current_period(plan, evaluator_factory, wrt_qids, dataslate.dates[t], base_evaluator, name_to_qid, )
        current_evaluator.iter_printer.header_message = \
            _create_header_message(vid, t, dataslate.dates[t], )
        dataslate.data[current_wrt_qids, t] = starter(dataslate.data, current_wrt_qids, t, )
        catch_missing(dataslate.data, t, )
        #
        init = current_evaluator.get_init_guess(dataslate.data, t, )
        root_final = _sp.optimize.root(
            current_evaluator.eval, init,
            args=(dataslate.data, t, None, ),
            jac=True,
            **root_settings,
        )
        #
        if not root_final.success:
            _catch_fail(root_final, when_fails, )
        current_evaluator.update(root_final.x, dataslate.data, t, )
        #
        current_evaluator.iter_printer.print_footer()
        current_evaluator.iter_printer.reset()


# REFACTOR
def _set_up_current_period(
    plan: _plans.PlanSimulate | None,
    evaluator_factory: Callable[..., _evaluators.PeriodEvaluator],
    wrt_qids: tuple[int, ...],
    current_period: _dates.Dater,
    base_evaluator: _evaluators.PeriodEvaluator,
    name_to_qid: dict[str, int],
    /,
) -> tuple[tuple[int, ...], _evaluators.PeriodEvaluator]:
    """
    """
    if plan is None:
        return wrt_qids, base_evaluator
    #
    names_exogenized = plan.get_names_exogenized_in_period(current_period, )
    names_endogenized = plan.get_names_endogenized_in_period(current_period, )
    if not names_exogenized and not names_endogenized:
        return wrt_qids, base_evaluator
    #
    if len(names_exogenized) != len(names_endogenized):
        raise _wrongdoings.IrisPieCritical(
            f"Number of exogenized quantities {len(names_exogenized)}"
            f" does not match number of endogenized quantities {len(names_endogenized)}"
            f" in period {current_period}"
        )
    #
    qids_exogenized = tuple(name_to_qid[name] for name in names_exogenized)
    qids_endogenized = tuple(name_to_qid[name] for name in names_endogenized)
    current_wrt_qids = tuple(sorted(set(wrt_qids).difference(qids_exogenized).union(qids_endogenized)))
    current_evaluator = evaluator_factory(current_wrt_qids, )
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
    dataslate: _dataslate.Dataslate,
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
    current_period = dataslate.dates[t]
    shift = 0
    for qid in _np.flatnonzero(missing):
        when_missing_stream.add(
            f"{qid_to_name[qid]}[{current_period+shift}]"
            f" when simulating {current_period}"
        )


def _catch_fail(
    root_final: _sp.optimize.OptimizeResult,
    when_fails: Literal["error", "warning", "silent"],
) -> None:
    """
    """
    raise ValueError()


def _create_header_message(
    vid: str,
    t: int,
    current_period: int,
) -> str:
    """
    """
    return f"[Variant {vid}][Period {current_period}]"


_ITER_STARTER = {
    "previous_period": _start_iter_from_previous_period,
    "data": _start_iter_from_data,
    "first_order": _start_iter_from_first_order,
}


_DEFAULT_ROOT_SETTINGS = {
    "method": "hybr",
    "tol": 1e-12,
}

