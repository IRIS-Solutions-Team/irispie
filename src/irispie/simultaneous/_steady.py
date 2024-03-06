"""
"""


#[
from __future__ import annotations
from typing import TYPE_CHECKING

from numbers import (Number, )
from collections.abc import (Iterable, Callable, )
from typing import (TypeAlias, Any, Literal, )
import copy as _copy
import functools as _ft
import itertools as _it
import numpy as _np
import scipy as _sp
import dataclasses as _dc

from .. import equations as _eq
from .. import quantities as _quantities
from .. import wrongdoings as _wrongdoings
from ..fords import steadiers as _fs
from ..evaluators import steady as _evaluators

from . import variants as _variants
from . import _flags
#]


@_dc.dataclass
class PlannableForSteady:
    """
    Implement PlannableProtocol
    """
    #[

    can_be_exogenized: tuple[str, ...]
    can_be_endogenized: tuple[str, ...]
    can_be_fixed_level: tuple[str, ...]
    can_be_fixed_change: tuple[str, ...]

    def __init__(
        self,
        model,
        /,
    ) -> None:
        """
        """
        generate = _quantities.generate_quantity_names_by_kind
        #
        self.can_be_exogenized = tuple(generate(
            model._invariant.quantities,
            _quantities.QuantityKind.ENDOGENOUS_VARIABLE,
        ))
        #
        self.can_be_endogenized = tuple(generate(
            model._invariant.quantities,
            _quantities.QuantityKind.PARAMETER,
        ))
        #
        self.can_be_fixed_level = tuple(generate(
            model._invariant.quantities,
            _quantities.QuantityKind.ENDOGENOUS_VARIABLE,
        ))
        #
        self.can_be_fixed_change = self.can_be_fixed_level

    #]


class SteadyInlay:
    """
    """
    #[

    def steady(
        self,
        /,
        **kwargs,
    ) -> list[dict]:
        """
        Calculate steady state for each Variant within this model
        """
        model_flags = _flags.Flags.update_from_kwargs(self.get_flags(), **kwargs)
        solver = self._choose_steady_solver(model_flags.is_linear, model_flags.is_flat, )
        info = [
            solver(v, model_flags, vid, **kwargs, )
            for vid, v in enumerate(self._variants, )
        ]
        return info

    def _steady_linear(
        self, 
        variant: _variants.Variant,
        model_flags: _flags.Flags,
        vid: int,
        /,
        *,
        algorithm: Callable,
        **kwargs,
    ) -> dict[str, Any]:
        """
        """
        #
        # Calculate first-order system for steady equations for this variant
        system = self._systemize(
            variant,
            self._invariant.steady_descriptor,
            model_flags,
        )
        #
        #=======================================================================
        # Core algorithm: Calculate steady state for this variant
        Xi, Y, dXi, dY = algorithm(system)
        levels = _np.hstack(( Xi.flat, Y.flat )).flatten()
        changes = _np.hstack(( dXi.flat, dY.flat )).flatten()
        #=======================================================================
        #
        # From first-order state space, extract only tokens with zero shift
        tokens = tuple(_it.chain(
            self._invariant.steady_descriptor.system_vectors.transition_variables,
            self._invariant.steady_descriptor.system_vectors.measurement_variables,
        ))
        #
        # True for tokens with zero shift, e.g. [True, False, True, ... ]
        zero_shift_index = [ not t.shift for t in tokens ]
        #
        # Extract steady levels for quantities with zero shift
        levels = levels[zero_shift_index]
        changes = changes[zero_shift_index]
        qids = tuple(t.qid for t in _it.compress(tokens, zero_shift_index))
        #
        # Delogarithmize when needed
        qid_to_logly = self.create_qid_to_logly()
        levels = _apply_delog_on_vector(levels, qids, qid_to_logly)
        changes = _apply_delog_on_vector(changes, qids, qid_to_logly)
        #
        variant.update_levels_from_array(levels, qids, )
        variant.update_changes_from_array(changes, qids, )
        #
        info = {}
        return info

    _steady_linear_flat = _ft.partialmethod(
        _steady_linear,
        algorithm=_fs.solve_steady_linear_flat,
    )

    _steady_linear_nonflat = _ft.partialmethod(
        _steady_linear,
        algorithm=_fs.solve_steady_linear_nonflat,
    )

    def _steady_nonlinear(
        self,
        variant: _variants.Variant,
        model_flags: _flags.Flags,
        vid: int,
        /,
        evaluator_class: type,
        *,
        fix: Iterable[str] | None = None,
        fix_levels: Iterable[str] | None = None,
        fix_changes: Iterable[str] | None = None,
        root_settings: dict[str, Any] | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        """
        root_settings = (
            _DEFAULT_ROOT_SETTINGS
            if root_settings is None
            else _DEFAULT_ROOT_SETTINGS | root_settings
        )
        #
        # REFACTOR: Plan.steady
        #
        wrt_equations = self.get_steady_equations()
        fixable_quantities = \
            self.get_quantities(kind=_quantities.QuantityKind.ENDOGENOUS_VARIABLE, )
        wrt_qids_levels = \
            _resolve_qids_fixed(fixable_quantities, fix, fix_levels, )
        wrt_qids_changes = \
            _resolve_qids_fixed(fixable_quantities, fix, fix_changes, )
        #
        all_quantities = self.get_quantities()
        steady_evaluator = evaluator_class(
            wrt_qids_levels,
            wrt_qids_changes,
            wrt_equations,
            all_quantities,
            variant,
            iter_printer_settings=iter_printer_settings,
        )
        block = 0
        header_message = f"[Variant {vid}][Block {block}]"
        steady_evaluator.iter_printer.header_message = header_message
        #
        root_final = _sp.optimize.root(
           steady_evaluator.eval,
           steady_evaluator.get_init_guess(),
           jac=True,
           **root_settings,
        )
        #
        steady_evaluator.iter_printer.print_footer()
        #
        final_guess = root_final.x
        func_norm = _sp.linalg.norm(root_final.fun, 2, )
        success = root_final.success and func_norm < root_settings["tol"]
        #
        if not success:
            raise _wrongdoings.IrisPieError(
                "Steady state calculations failed to converge"
            )
        #
        levels, wrt_qids_levels = steady_evaluator.extract_levels(final_guess, )
        variant.update_levels_from_array(levels, wrt_qids_levels, )
        changes, wrt_qids_changes = steady_evaluator.extract_changes(final_guess, )
        variant.update_changes_from_array(changes, wrt_qids_changes, )
        #
        info = {
            "success": success,
            "root_final": root_final,
        }
        return info

    _steady_nonlinear_flat = _ft.partialmethod(
        _steady_nonlinear,
        evaluator_class=_evaluators.FlatSteadyEvaluator,
    )

    _steady_nonlinear_nonflat = _ft.partialmethod(
        _steady_nonlinear,
        evaluator_class=_evaluators.NonflatSteadyEvaluator,
    )

    def _choose_steady_solver(
        self,
        is_linear: bool,
        is_flat: bool,
        /,
    ) -> Callable:
        """
        Choose steady solver depending on linear and flat flags
        """
        match (is_linear, is_flat, ):
            case (False, False, ):
                return self._steady_nonlinear_nonflat
            case (False, True, ):
                return self._steady_nonlinear_flat
            case (True, False, ):
                return self._steady_linear_nonflat
            case (True, True, ):
                return self._steady_linear_flat

    def check_steady(
        self,
        /,
        equation_switch: Literal["steady", "dynamic"] = "dynamic",
        when_fails: _wrongdoings.HOW = "error",
        tolerance: float = 1e-12,
    ) -> tuple[bool, tuple[dict, ...]]:
        """
        Verify currently assigned steady state in dynamic or steady equations for each variant within this model
        """
        qid_to_logly = self.create_qid_to_logly()
        equator = self._choose_plain_equator(equation_switch)
        steady_arrays = (
            v.create_steady_array(
                qid_to_logly,
                num_columns=equator.min_num_columns + 1,
                shift_in_first_column=equator.min_shift,
            ) for v in self._variants
        )
        #
        # REFACTOR
        #
        t_zero = -equator.min_shift
        dis = [ 
            _np.hstack((
                equator.eval(x, t_zero, x[:, t_zero]),
                equator.eval(x, t_zero+1, x[:, t_zero+1]),
            ))
            for x in steady_arrays
        ]
        #
        # REFACTOR
        #
        max_abs_dis = [ _np.max(_np.abs(d)) for d in dis ]
        status = [ d < tolerance for d in max_abs_dis ]
        all_status = all(status)
        if not all_status:
            message = "Invalid steady state"
            _wrongdoings.raise_as(when_fails, message)
        details = [
            {"discrepancies": d, "max_abs_discrepancy": m, "is_valid": s}
            for d, m, s in zip(dis, max_abs_dis, status)
        ]
        return all_status, details

    #
    # ===== Implement PlannableProtocol =====
    # This protocol is used to create Plan objects
    #

    def get_plannable_for_steady(self, /) -> PlannableForSteady:
        return PlannableForSteady(self, )

    #]


def _apply_delog_on_vector(
    vector: _np.ndarray,
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> _np.ndarray:
    """
    Delogarithmize the elements of numpy vector that have True log-status
    """
    #[
    logly_index = [ qid_to_logly[qid] for qid in qids ]
    vector = _np.copy(vector)
    if any(logly_index):
        vector[logly_index] = _np.exp(vector[logly_index])
    return vector
    #]


def _resolve_qids_fixed(
    fixable_quantities,
    fix: Iterable[str] | str | None,
    fix_spec: Iterable[str] | str | None,
) -> tuple[int, ...]:
    """
    """
    #[
    if fix is None:
        fix = ()
    elif isinstance(fix, str):
        fix = (fix, )
    if fix_spec is None:
        fix_spec = ()
    elif isinstance(fix_spec, str):
        fix_spec = (fix_spec, )
    fix = tuple(i for i in set(fix) | set(fix_spec) if not i.startswith("!"))
    qids_fixed, invalid_names = \
        _quantities.lookup_qids_by_name(fixable_quantities, fix, )
    if invalid_names:
        raise _wrongdoings.IrisPieError(
            (f"Cannot fix these names:", ) + invalid_names
        )
    wrt_qids = tuple(
        q.id
        for q in fixable_quantities
        if q.id not in qids_fixed
    )
    return wrt_qids
    #]


_DEFAULT_ROOT_SETTINGS = {
    "method": "lm",
    "tol": 1e-12,
}

