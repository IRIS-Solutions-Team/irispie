"""
"""


#[
from __future__ import annotations
from typing import TYPE_CHECKING

from numbers import (Number, )
from collections import (namedtuple, )
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
from .. import has_variants as _has_variants
from .. import wrongdoings as _wrongdoings
from ..fords import steadiers as _fs
from ..evaluators import steady as _evaluators
from ..plans.steady_plans import (SteadyPlan, )

from . import variants as _variants
from . import _flags
#]


_Wrts = namedtuple("Wrt", (
    "equations",
    "level_names",
    "change_names",
    "any_names",
    "level_qids",
    "change_qids",
    "any_qids",
))


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


class Inlay:
    """
    """
    #[

    def steady(
        self,
        /,
        unpack_singleton: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> dict | list[dict]:
        """
        Calculate steady state for each Variant within this model
        """
        model_flags = self.resolve_flags(**kwargs, )
        solver = self._choose_steady_solver(model_flags.is_linear, model_flags.is_flat, )
        out_info = []
        for vid, v in enumerate(self._variants, ):
            out_info_v = solver(v, model_flags, vid, **kwargs, )
            out_info.append(out_info_v, )
        #
        if return_info:
            out_info = _has_variants.unpack_singleton(
                out_info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return out_info
        else:
            return

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
        self.delogarithmize(levels, changes, )
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
        plan: SteadyPlan | None = None,
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
        wrts = self.resolve_steady_wrts(plan, is_flat=model_flags.is_flat, )

        #
        all_quantities = self.get_quantities()
        steady_evaluator = evaluator_class(
            wrts.level_qids,
            wrts.change_qids,
            wrts.equations,
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
        levels, wrt_level_qids = steady_evaluator.extract_levels(final_guess, )
        variant.update_levels_from_array(levels, wrt_level_qids, )
        changes, wrt_change_qids = steady_evaluator.extract_changes(final_guess, )
        variant.update_changes_from_array(changes, wrt_change_qids, )
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

    def resolve_steady_wrts(
        self,
        plan: SteadyPlan,
        is_flat: bool,
    ) -> _Wrts:
        """
        """
        wrt_equations = self.get_steady_equations()
        #
        name_to_qid = self.create_name_to_qid()
        plannable = self.get_steady_plannable(is_flat=is_flat, )
        wrt_names = set(plannable.can_be_exogenized)
        if plan is None or plan.is_empty:
            exogenized_names = ()
            endogenized_names = ()
            fixed_level_names = ()
            fixed_change_names = ()
        else:
            exogenized_names = plan.get_exogenized_names()
            endogenized_names = plan.get_endogenized_names()
            fixed_level_names = plan.get_fixed_level_names()
            fixed_change_names = plan.get_fixed_change_names()
        #
        wrt_names = (wrt_names - set(exogenized_names)) | set(endogenized_names)
        wrt_level_names = (wrt_names - set(fixed_level_names))
        wrt_change_names = (wrt_names - set(fixed_change_names)) if not is_flat else ()
        wrt_any_names = tuple(sorted(set(wrt_level_names) | set(wrt_change_names)))
        #
        wrt_level_qids = tuple(sorted([name_to_qid[name] for name in wrt_level_names]))
        wrt_change_qids = tuple(sorted([name_to_qid[name] for name in wrt_change_names]))
        wrt_any_qids = tuple(sorted([name_to_qid[name] for name in wrt_any_names]))
        #
        return _Wrts(
            equations=wrt_equations,
            level_names=wrt_level_names,
            change_names=wrt_change_names,
            any_names=wrt_any_names,
            level_qids=wrt_level_qids,
            change_qids=wrt_change_qids,
            any_qids=wrt_any_qids,
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
        return_info: bool = False,
    ) -> tuple[bool, tuple[dict, ...]]:
        """
        Verify currently assigned steady state in dynamic or steady equations for each variant within this model
        """
        qid_to_logly = self.create_qid_to_logly()
        equator = self._choose_plain_equator(equation_switch, )
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
        #
        if not return_info:
            return all_status
        else:
            info = [
                {"discrepancies": d, "max_abs_discrepancy": m, "is_valid": s}
                for d, m, s in zip(dis, max_abs_dis, status)
            ]
            return all_status, info

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
) -> None:
    """
    Delogarithmize the elements of numpy vector that have True log-status
    """
    #[
    #]


_DEFAULT_ROOT_SETTINGS = {
    "method": "lm",
    "tol": 1e-12,
}

