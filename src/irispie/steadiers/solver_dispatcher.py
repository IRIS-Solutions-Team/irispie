"""
Facades for steady-state solvers
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp
import neqs as _ne

from .evaluators import SteadyEvaluator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    SolverType = Literal["neqs_levenberg", "neqs_firefly", "scipy_root", ]

#]


#===============================================================================


_DEFAULT_SCIPY_ROOT_SETTINGS = {
    "method": "lm",
    "tol": 1e-12,
}


def scipy_root(
    steady_evaluator: SteadyEquator,
    maybelog_init_guess: _np.ndarray,
    *,
    solver_settings: dict[str, Any] | None = None,
) -> tuple[bool, _np.ndarray, Any, ]:
    """
    """
    solver_settings = solver_settings or {}
    solver_settings = _DEFAULT_SCIPY_ROOT_SETTINGS | solver_settings
    #
    root_final = _sp.optimize.root(
        steady_evaluator.eval,
        maybelog_init_guess,
        jac=True,
        **solver_settings,
    )
    #
    steady_evaluator.iter_printer.print_footer()
    maybelog_final_guess = root_final.x
    func_norm = _sp.linalg.norm(root_final.fun, 2, )
    success = root_final.success and func_norm < solver_settings["tol"]
    # TODO: create exit status based on root_final
    exit_status = None
    return maybelog_final_guess, success, exit_status,


#===============================================================================


_DEFAULT_NEQS_LEVENBERG_SETTINGS = {
    "step_tolerance": float("inf"),
    "func_tolerance": 1e-14,
}


def neqs_levenberg(
    steady_evaluator: SteadyEquator,
    maybelog_init_guess: _np.ndarray,
    *,
    solver_settings: dict[str, Any] | None = None,
) -> tuple[bool, _np.ndarray, Any, ]:
    """
    """
    solver_settings = solver_settings or {}
    solver_settings = _DEFAULT_NEQS_LEVENBERG_SETTINGS | solver_settings
    maybelog_final_guess, exit_status = _ne.levenberg(
        eval_func=steady_evaluator.eval_func,
        eval_jacob=steady_evaluator.eval_jacob,
        init_guess=maybelog_init_guess,
        iter_printer=steady_evaluator.iter_printer,
        **solver_settings,
    )
    success = exit_status.is_success
    return maybelog_final_guess, success, exit_status,


#===============================================================================


_DEFAULT_NEQS_FIREFLY_SETTINGS = {
}


def neqs_firefly(
    steady_evaluator: SteadyEquator,
    maybelog_init_guess: _np.ndarray,
    *,
    solver_settings: dict[str, Any] | None = None,
) -> tuple[bool, _np.ndarray, Any, ]:
    """
    """
    solver_settings = solver_settings or {}
    solver_settings = _DEFAULT_NEQS_FIREFLY_SETTINGS | solver_settings
    #
    maybelog_final_guess = _ne.firefly(
        eval_func=steady_evaluator.eval_func,
        init_guess=maybelog_init_guess,
        iter_printer=steady_evaluator.iter_printer,
        solver_settings=solver_settings,
    )
    #
    success = True
    exit_status = None
    return maybelog_final_guess, success, exit_status,


