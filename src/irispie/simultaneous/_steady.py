"""
"""


#[

from __future__ import annotations

from collections import namedtuple
import functools as _ft
import itertools as _it
import numpy as _np
import scipy as _sp
import neqs as _ne

from .. import equations as _equations
from ..incidences import blazer as _blazer
from ..incidences.blazer import Block, HumanBlock
from .. import has_variants as _has_variants
from .. import wrongdoings as _wrongdoings
from ..fords import steadiers as _fs
from ..steadiers import evaluators as _evaluators
from ..steadiers import solver_dispatcher as _solver_dispatcher
from ..equations import ENDOGENOUS_EQUATION
from ..quantities import QuantityKind

from . import _flags

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..plans.steady_plans import SteadyPlan
    from ..equations import Equation
    from ._variants import Variant
    from collections.abc import Iterable, Callable
    from typing import Any, Literal, NoReturn
    from ..steadiers.solver_dispatcher import SolverType

#]


_Wrt = namedtuple("Wrt", (
    "equations",
    "eids",
    "names",
    "fixed_level_names",
    "fixed_change_names",
    "qids",
    "fixed_level_qids",
    "fixed_change_qids",
))


_STEADY_EQUATION_SOLVED = ENDOGENOUS_EQUATION

_DEFAULT_SOLVER = "neqs_levenberg"


class Inlay:
    """
    """
    #[

    def steady(
        self,
        unpack_singleton: bool = True,
        update_steady_autovalues: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> dict | list[dict]:
        """
        Calculate steady state for each Variant within this model
        """
        model_flags = self.resolve_flags(**kwargs, )
        steady_solver = self._choose_steady_solver(model_flags.is_linear, model_flags.is_flat, )
        out_info = []
        for vid, v in enumerate(self._variants, ):
            out_info_v = steady_solver(v, model_flags, vid, **kwargs, )
            out_info.append(out_info_v, )
        if update_steady_autovalues:
            self.update_steady_autovalues()
        if return_info:
            return _has_variants.unpack_singleton(
                out_info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )

    def _steady_linear(
        self,
        variant: Variant,
        model_flags: _flags.Flags,
        vid: int,
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
        variant: Variant,
        model_flags: _flags.Flags,
        vid: int,
        evaluator_class: type,
        *,
        plan: SteadyPlan | None = None,
        split_into_blocks: bool | None = None,
        solver: SolverType = _DEFAULT_SOLVER,
        solver_settings: dict[str, Any] | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        """
        qid_to_kind = self.create_qid_to_kind()
        qid_to_name = self.create_qid_to_name()
        #
        split_into_blocks = _resolve_split_into_blocks(split_into_blocks, plan, )
        wrt = self._resolve_steady_wrt(plan, is_flat=model_flags.is_flat, )
        if split_into_blocks:
            im = _calculate_steady_incidence_matrix(wrt.equations, wrt.qids, )
            blocks = _blazer.blaze(im, wrt.eids, wrt.qids, )
        else:
            blocks = (_blazer.Block(wrt.eids, wrt.qids), )
        num_blocks = len(blocks)
        #
        all_quantities = self.get_quantities()
        all_equations = self.get_steady_equations(kind=_STEADY_EQUATION_SOLVED, )
        human_blocks \
            = _blazer.human_blocks_from_blocks \
            (blocks, all_equations, all_quantities, )
        #
        # Prepare the solver and iter printer settings
        #
        solve = getattr(_solver_dispatcher, solver)
        create_solver_settings = getattr(_solver_dispatcher, "create_solver_settings_for_" + solver, )
        solver_settings = create_solver_settings(
            user_solver_settings=solver_settings,
            self_equality_tolerance=self.get_tolerance("equality", ),
        )
        iter_printer_settings = iter_printer_settings or {}
        iter_printer_settings["norm_order"] = solver_settings["norm_order"]
        #
        #
        info_v = {
            "success": None,
            "blocks": [],
        }
        #=======================================================================
        for bid, (block, human_block) in enumerate(zip(blocks, human_blocks, ), ):
            #
            info_b = {
                "success": None,
                "exit_status": None,
                "equations": human_block.equations,
                "quantities": human_block.quantities,
            }
            custom_header = f"[Variant {vid}][Block {bid}]"
            #
            block_level_qids = tuple(sorted(set(block.qids) - set(wrt.fixed_level_qids)))
            block_change_qids = tuple(sorted(set(block.qids) - set(wrt.fixed_change_qids)))
            block_equations = tuple(wrt.equations[eid] for eid in block.eids)
            #
            has_no_qids = (not block_level_qids) and (not block_change_qids)
            has_no_equations = not block_equations
            #
            if has_no_qids or has_no_equations:
                info_b["success"] = True
                info_b["exit_status"] = _ne.ExitStatus.NO_SOLVER_NEEDED
                info_v["blocks"].append(info_b, )
                print(f"\n-Skipping {custom_header}")
                continue
            #
            steady_evaluator = evaluator_class(
                block_level_qids,
                block_change_qids,
                block_equations,
                all_quantities,
                variant,
                context=self._invariant._context,
                iter_printer_settings=iter_printer_settings | {"custom_header": custom_header},
            )
            #
            maybelog_init_guess = steady_evaluator.get_init_guess()
            maybelog_final_guess, success, exit_status = solve(
                steady_evaluator,
                maybelog_init_guess,
                solver_settings=solver_settings,
            )
            steady_evaluator.final_guess = maybelog_final_guess
            #
            if not success:
                _throw_block_error(human_block, custom_header, )
            #
            # Update variant with steady levels and changes
            _update_variant_with_final_guess(variant, steady_evaluator, qid_to_kind, qid_to_name, )
            #
            info_b["success"] = success
            info_b["exit_status"] = exit_status
            info_v["blocks"].append(info_b, )
        #=======================================================================
        #
        info_v["success"] = all(i["success"] for i in info_v["blocks"])
        return info_v

    _steady_nonlinear_flat = _ft.partialmethod(
        _steady_nonlinear,
        evaluator_class=_evaluators.FlatSteadyEvaluator,
    )

    _steady_nonlinear_nonflat = _ft.partialmethod(
        _steady_nonlinear,
        evaluator_class=_evaluators.NonflatSteadyEvaluator,
    )

    def _resolve_steady_wrt(
        self,
        plan: SteadyPlan,
        is_flat: bool,
    ) -> _Wrt:
        """
        """
        wrt_equations = self.get_steady_equations(kind=_STEADY_EQUATION_SOLVED, )
        wrt_eids = tuple(e.id for e in wrt_equations)
        #
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
            fixed_change_names = plan.get_fixed_change_names() + endogenized_names
        #
        wrt_names = (wrt_names - set(exogenized_names)) | set(endogenized_names)
        # wrt_level_names = (wrt_names - set(fixed_level_names))
        # wrt_change_names = (wrt_names - set(fixed_change_names)) if not is_flat else ()
        # wrt_any_names = wrt_level_names | wrt_change_names
        #
        name_to_qid = self.create_name_to_qid()
        # wrt_level_qids = tuple(sorted([name_to_qid[name] for name in wrt_level_names]))
        # wrt_change_qids = tuple(sorted([name_to_qid[name] for name in wrt_change_names]))
        def get_sorted_qids(names: Iterable[str], ) -> tuple[int, ...]:
            return tuple(sorted([name_to_qid[name] for name in names]))
        wrt_qids = get_sorted_qids(wrt_names, )
        wrt_fixed_level_qids = get_sorted_qids(fixed_level_names, )
        wrt_fixed_change_qids = get_sorted_qids(fixed_change_names, )
        #
        # Make order of names consistent with qids
        qid_to_name = self.create_qid_to_name()
        # wrt_level_names = tuple(qid_to_name[qid] for qid in wrt_level_qids)
        # wrt_change_names = tuple(qid_to_name[qid] for qid in wrt_change_qids)
        wrt_names = tuple(qid_to_name[qid] for qid in wrt_qids)
        wrt_fixed_level_names = tuple(qid_to_name[qid] for qid in wrt_fixed_level_qids)
        wrt_fixed_change_names = tuple(qid_to_name[qid] for qid in wrt_fixed_change_qids)
        #
        return _Wrt(
            equations=wrt_equations,
            eids=wrt_eids,
            names=wrt_names,
            fixed_level_names=wrt_fixed_level_names,
            fixed_change_names=wrt_fixed_change_names,
            qids=wrt_qids,
            fixed_level_qids=wrt_fixed_level_qids,
            fixed_change_qids=wrt_fixed_change_qids,
        )

    def _choose_steady_solver(
        self,
        is_linear: bool,
        is_flat: bool,
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
        equation_switch: Literal["steady", "dynamic"] = "dynamic",
        when_fails: _wrongdoings.HOW = "error",
        tolerance: float | None = None,
        return_info: bool = False,
        unpack_singleton: bool = True,
    ) -> NoReturn | tuple[bool, tuple[dict, ...]]:
        """
        Verify currently assigned steady state in dynamic or steady equations for each variant within this model
        """
        tolerance = tolerance or self.get_tolerance("equality", )
        qid_to_logly = self.create_qid_to_logly()
        equations = getattr(self._invariant, f"{equation_switch}_equations", )
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
        discrepancies = []
        for x in steady_arrays:
            at_t_zero = equator.eval(x, t_zero, )
            at_t_zero_plus_one = equator.eval(x, t_zero + 1, )
            discrepancies.append(_np.hstack((
                _np.array(at_t_zero).reshape(-1, 1),
                _np.array(at_t_zero_plus_one).reshape(-1, 1),
            )))
        #
        # REFACTOR
        #
        fail_stream = _wrongdoings.create_stream(
            when_fails,
            "Steady state discrepancies in these equations",
        )
        status = []
        failed_discrepancies = []
        failed_equations = []
        for vid, d, in enumerate(discrepancies, ):
            where = _np.any(_np.abs(d) > tolerance, axis=1, ).nonzero()[0].tolist()
            for j in where:
                fail_stream.add(f"[Variant {vid}] {equations[j].human}: {d[j, :].tolist()}")
            status.append(not where)
            failed_discrepancies.append(d[where, :], )
            failed_equations.append(tuple(equations[j].human for j in where), )
        overall_status = all(status)
        #
        fail_stream._raise()
        #
        if not return_info:
            return overall_status
        #
        info = [
            {"discrepancies": d, "failed_discrepancies": f, "failed_equations": e, }
            for d, f, e in zip(discrepancies, failed_discrepancies, failed_equations, )
        ]
        info = _has_variants.unpack_singleton(
            info, self.is_singleton,
            unpack_singleton=unpack_singleton,
        )
        return overall_status, info,


    def split_into_blocks(
        self,
        plan: SteadyPlan,
        **kwargs,
    ) -> tuple[HumanBlock, ...]:
        r"""
        """
        model_flags = self.resolve_flags(**kwargs, )
        wrt = self._resolve_steady_wrt(plan, is_flat=model_flags.is_flat, )
        im = _calculate_steady_incidence_matrix(wrt.equations, wrt.qids, )
        blocks = _blazer.blaze(im, wrt.eids, wrt.qids, )
        equations = self._invariant.steady_equations
        quantities = self._invariant.quantities
        return tuple(
            HumanBlock(i, equations, quantities, )
            for i in blocks
        )

    #]


def _apply_delog_on_vector(
    vector: _np.ndarray,
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
) -> None:
    """
    Delogarithmize the elements of numpy vector that have True log-status
    """
    #[
    #]


def _calculate_steady_incidence_matrix(
    equations: Iterable[Equation],
    wrt_any_qids: Iterable[int],
) -> _np.ndarray:
    """
    """
    qid_to_column = { qid: column for column, qid in enumerate(wrt_any_qids) }
    def token_within_quantities(tok: Token, ) -> bool:
        return qid_to_column.get(tok.qid, None)
        #
    return _equations.calculate_incidence_matrix(
        equations,
        len(wrt_any_qids),
        token_within_quantities,
    )


def _resolve_split_into_blocks(
    split_into_blocks: bool | None,
    plan: SteadyPlan | None,
) -> bool:
    """
    """
    #[
    if split_into_blocks is not None:
        return split_into_blocks
    if plan is None:
        return True
    any_fix = (
        plan.any_in_register("fixed_level", )
        or plan.any_in_register("fixed_change", )
    )
    return not any_fix
    #]


def _update_variant_with_final_guess(variant, steady_evaluator, qid_to_kind, qid_to_name, ) -> None:
    """
    """
    #[
    final_guess = steady_evaluator.final_guess
    #
    levels, wrt_level_qids = steady_evaluator.extract_levels(final_guess, )
    variant.update_levels_from_array(levels, wrt_level_qids, )
    #
    changes, wrt_change_qids = steady_evaluator.extract_changes(final_guess, )
    #
    # Filter out non-loggable quantities, i.e. endogenized parameters (only loggable
    # quantities can have steady-state change)
    changes_to_update = []
    wrt_change_qids_to_update = []
    for change, qid in zip(changes, wrt_change_qids, ):
        if qid_to_kind[qid] not in QuantityKind.LOGGABLE_VARIABLE:
            continue
        changes_to_update.append(change)
        wrt_change_qids_to_update.append(qid)
    variant.update_changes_from_array(changes_to_update, wrt_change_qids_to_update, )
    #]


def _throw_block_error(human_block, custom_header: str, ) -> NoReturn:
    """
    """
    #[
    message = (
        f"Steady state calculations failed to converge in {custom_header}",
        *human_block.equations,
        "Solved for " + " ".join(human_block.quantities, ),
    )
    raise _wrongdoings.IrisPieError(message, )
    #]


