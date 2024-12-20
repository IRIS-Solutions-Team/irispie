"""
Handles steady-state calculations for simultaneous models.
"""

#[
from __future__ import annotations

from collections import namedtuple
import functools as _ft
import itertools as _it
import numpy as _np
import scipy as _sp
import neqs as _nq

from .. import equations as _equations
from ..incidences import blazer as _blazer
from .. import has_variants as _has_variants
from .. import wrongdoings as _wrongdoings
from ..fords import steadiers as _fs
from ..steadiers import evaluators as _evaluators
from ..sources import LOGGABLE_VARIABLE
from ..equations import ENDOGENOUS_EQUATION

from . import _flags

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..plans.steady_plans import SteadyPlan
    from ..equations import Equation
    from ._variants import Variant
    from collections.abc import Iterable, Callable
    from typing import Any, Literal, NoReturn
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


class Inlay:
    r"""
    ................................................................................
    ==Class: Inlay==

    Facilitates steady-state calculations for simultaneous models. It manages the 
    logic for linear and nonlinear steady-state solvers, provides utilities for 
    data validation, and ensures consistency across variants.

    Attributes:
        - `_variants`: List of model variants.
        - `_invariant`: Model-wide configuration and constraints.
    ................................................................................
    """
    #[

    def steady(
        self,
        /,
        unpack_singleton: bool = True,
        update_steady_autovalues: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> dict | list[dict]:
        r"""
        ................................................................................
        ==Method: steady==

        Calculates the steady state for each variant within the model. Supports both 
        linear and nonlinear solvers, automatically selecting the appropriate method.

        ### Input arguments ###
        ???+ input "unpack_singleton: bool = True"
            If `True`, unpacks results when only a single variant is simulated.
        ???+ input "update_steady_autovalues: bool = True"
            Whether to update steady autovalues after calculations.
        ???+ input "return_info: bool = False"
            If `True`, returns detailed information about the calculation process.
        ???+ input "**kwargs"
            Additional arguments to configure steady-state calculations.

        ### Returns ###
        ???+ returns "dict | list[dict]"
            A dictionary or list of dictionaries containing steady-state information 
            for each variant.

        ### Example ###
        ```python
            steady_info = model.steady(return_info=True)
        ```
        ................................................................................
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
        /,
        *,
        algorithm: Callable,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Method: _steady_linear==

        Solves steady-state equations for a linear model variant. The solver uses 
        first-order state space to calculate steady-state levels and changes.

        ### Input arguments ###
        ???+ input "variant: Variant"
            The model variant for which the steady state is being calculated.
        ???+ input "model_flags: _flags.Flags"
            Flags indicating model configuration (e.g., linearity).
        ???+ input "vid: int"
            The index of the variant being processed.
        ???+ input "algorithm: Callable"
            The algorithm used to solve the steady-state equations.
        ???+ input "**kwargs"
            Additional arguments passed to the solver.

        ### Returns ###
        ???+ returns "dict[str, Any]"
            Information about the calculation process and results.

        ### Example ###
        ```python
            info = model._steady_linear(variant, flags, vid, algorithm=custom_solver)
        ```
        ................................................................................
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
        /,
        evaluator_class: type,
        *,
        plan: SteadyPlan | None = None,
        split_into_blocks: bool | None = None,
        root_settings: dict[str, Any] | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
        ................................................................................
        ==Method: _steady_nonlinear==

        Solves steady-state equations for a nonlinear model variant. The solver uses 
        block-based or full-system approaches based on the model configuration and 
        the provided plan.

        ### Input arguments ###
        ???+ input "variant: Variant"
            The model variant for which the steady state is being calculated.
        ???+ input "model_flags: _flags.Flags"
            Flags indicating model configuration (e.g., linearity, flat mode).
        ???+ input "vid: int"
            The index of the variant being processed.
        ???+ input "evaluator_class: type"
            The evaluator class to use for steady-state calculations (e.g., flat or nonflat).
        ???+ input "plan: SteadyPlan | None = None"
            A steady-state plan specifying exogenized or fixed variables.
        ???+ input "split_into_blocks: bool | None = None"
            Whether to split the equations into blocks. Defaults to automatic resolution.
        ???+ input "root_settings: dict[str, Any] | None = None"
            Settings for the root-finding algorithm.
        ???+ input "iter_printer_settings: dict[str, Any] | None = None"
            Settings for iterative print logs during calculations.
        ???+ input "**kwargs"
            Additional arguments passed to the solver.

        ### Returns ###
        ???+ returns "dict[str, Any]"
            Information about the calculation process and results.

        ### Example ###
        ```python
            info = model._steady_nonlinear(
                variant, flags, vid, evaluator_class=NonlinearEvaluator, plan=plan
            )
        ```
        ................................................................................
        """
        qid_to_kind = self.create_qid_to_kind()
        qid_to_name = self.create_qid_to_name()
        root_settings = (
            _DEFAULT_ROOT_SETTINGS
            if root_settings is None
            else _DEFAULT_ROOT_SETTINGS | root_settings
        )
        #
        split_into_blocks = _resolve_split_into_blocks(split_into_blocks, plan, )
        wrt = self._resolve_steady_wrt(plan, is_flat=model_flags.is_flat, )
        if split_into_blocks:
            im = _calculate_steady_incidence_matrix(wrt.equations, wrt.qids, )
            blocks = _blazer.blaze(im, wrt.eids, wrt.qids, )
        else:
            blocks = (_blazer._Block(wrt.eids, wrt.qids), )
        #
        all_quantities = self.get_quantities()
        all_equations = self.get_steady_equations(kind=_STEADY_EQUATION_SOLVED, )
        human_blocks \
            = _blazer.human_blocks_from_blocks \
            (blocks, all_equations, all_quantities, )
        #
        info = {
            "variant_success": None,
            "blocks": human_blocks,
            "block_success": [],
            "block_root_final": [],
        }
        #=======================================================================
        for bid, (block, human_block) in enumerate(zip(blocks, human_blocks, ), ):
            custom_header = f"[Variant {vid}][Block {bid}]"
            #
            block_level_qids = tuple(sorted(set(block.qids) - set(wrt.fixed_level_qids)))
            block_change_qids = tuple(sorted(set(block.qids) - set(wrt.fixed_change_qids)))
            block_equations = tuple(wrt.equations[eid] for eid in block.eids)
            #
            has_no_qids = (not block_level_qids) and (not block_change_qids)
            has_no_equations = not block_equations
            if has_no_qids or has_no_equations:
                info["block_success"].append(True, )
                info["block_root_final"].append(None, )
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
                iter_printer_settings=iter_printer_settings,
            )
            # neqs = False
            # if block_change_qids and len(block_change_qids) == len(block_level_qids) and len(block_level_qids) == len(block_equations):
            #     neqs = True
            # elif (not block_change_qids) and len(block_level_qids) == len(block_equations):
            #     neqs = True

            init_guess = steady_evaluator.get_init_guess()

            steady_evaluator.iter_printer.custom_header = custom_header
            root_final = _sp.optimize.root(
                steady_evaluator.eval,
                init_guess,
                jac=True,
                **root_settings,
            )
            steady_evaluator.iter_printer.print_footer()
            final_guess = root_final.x
            func_norm = _sp.linalg.norm(root_final.fun, 2, )
            success = root_final.success and func_norm < root_settings["tol"]

            # else:
            #     final_guess, exit_status = _nq.damped_newton(
            #         eval_func=steady_evaluator.eval_func,
            #         eval_jacob=steady_evaluator.eval_jacob,
            #         init_guess=init_guess,
            #     )
            #     success = exit_status.is_success

            steady_evaluator.final_guess = final_guess
            #
            #
            if not success:
                _throw_block_error(human_block, custom_header, )
            #
            # Update variant with steady levels and changes
            _update_variant_with_final_guess(variant, steady_evaluator, qid_to_kind, qid_to_name, )
            #
            info["block_success"].append(success, )
            info["block_root_final"].append(root_final, )
        #=======================================================================
        #
        info["variant_success"] = all(info["block_success"])
        #
        return info

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
        r"""
        ................................................................................
        ==Method: _resolve_steady_wrt==

        Resolves the "with respect to" (WRT) variables and equations for steady-state 
        calculations. Identifies the quantities to be considered for steady-state 
        updates, including fixed levels and changes.

        ### Input arguments ###
        ???+ input "plan: SteadyPlan"
            A steady-state plan specifying exogenized, fixed, and endogenized variables.
        ???+ input "is_flat: bool"
            Indicates whether the model operates in flat mode.

        ### Returns ###
        ???+ returns "_Wrt"
            A named tuple containing WRT variables, equations, and their respective 
            metadata.

        ### Example ###
        ```python
            wrt = model._resolve_steady_wrt(plan, is_flat=True)
        ```
        ................................................................................
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
        /,
    ) -> Callable:
        r"""
        ................................................................................
        ==Method: _choose_steady_solver==

        Selects the appropriate steady-state solver based on the model's linearity 
        and flat mode configuration.

        ### Input arguments ###
        ???+ input "is_linear: bool"
            Indicates whether the model is linear.
        ???+ input "is_flat: bool"
            Indicates whether the model operates in flat mode.

        ### Returns ###
        ???+ returns "Callable"
            The selected steady-state solver function.

        ### Example ###
        ```python
            solver = model._choose_steady_solver(is_linear=True, is_flat=False)
        ```
        ................................................................................
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
        unpack_singleton: bool = True,
    ) -> tuple[bool, tuple[dict, ...]]:
        r"""
        ................................................................................
        ==Method: check_steady==

        Verifies the steady-state solution for each model variant against the specified 
        equations (steady or dynamic). Identifies discrepancies exceeding the tolerance.

        ### Input arguments ###
        ???+ input "equation_switch: Literal[...] = 'dynamic'"
            Specifies whether to verify steady or dynamic equations.
        ???+ input "when_fails: _wrongdoings.HOW = 'error'"
            Determines the behavior when discrepancies are found (e.g., raise error).
        ???+ input "tolerance: float = 1e-12"
            The tolerance level for acceptable discrepancies.
        ???+ input "return_info: bool = False"
            If `True`, returns detailed information about discrepancies.
        ???+ input "unpack_singleton: bool = True"
            If `True`, unpacks results when only a single variant is checked.

        ### Returns ###
        ???+ returns "tuple[bool, tuple[dict, ...]]"
            A tuple containing:
            - A boolean indicating whether all variants passed the check.
            - Detailed discrepancy information (if requested).

        ### Example ###
        ```python
            status, info = model.check_steady(equation_switch="steady", return_info=True)
        ```
        ................................................................................
        """
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
        for vid, i in enumerate(discrepancies, ):
            where = _np.any(_np.abs(i) > tolerance, axis=1, ).nonzero()[0].tolist()
            for j in where:
                fail_stream.add(f"[Variant {vid}] {equations[j].human}")
            status.append(not where)
        fail_stream._raise()
        all_status = all(status)
        #
        if return_info:
            info = [
                {"discrepancies": d, }
                for d in discrepancies
            ]
            info = _has_variants.unpack_singleton(
                info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return all_status, info
        else:
            return all_status

    #]


def _apply_delog_on_vector(
    vector: _np.ndarray,
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> None:
    r"""
    ................................................................................
    ==Function: _apply_delog_on_vector==

    Applies an exponential transformation (delogarithmization) to specified elements 
    of a numerical vector. The transformation affects only the indices specified in 
    `logly_indexes`.

    ### Input arguments ###
    ???+ input "values: _np.ndarray"
        A numerical vector whose elements are to be transformed.
    ???+ input "logly_indexes: tuple[int, ...]"
        Indices of the vector elements to transform.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _apply_delog_on_vector(values, logly_indexes=(0, 2, 4))
    ```
    ................................................................................
    """
    #[
    #]


def _calculate_steady_incidence_matrix(
    equations: Iterable[Equation],
    wrt_any_qids: Iterable[int],
    /,
) -> _np.ndarray:
    r"""
    ................................................................................
    ==Function: _calculate_steady_incidence_matrix==

    Computes the incidence matrix for steady-state equations, mapping variables 
    (quantities) to the equations in which they are involved.

    ### Input arguments ###
    ???+ input "wrt_equations: tuple[Equation, ...]"
        The steady-state equations to analyze.
    ???+ input "wrt_qids: tuple[int, ...]"
        Quantity IDs (QIDs) to include in the incidence analysis.

    ### Returns ###
    ???+ returns "_np.ndarray"
        A binary incidence matrix indicating variable-equation relationships.

    ### Example ###
    ```python
        incidence_matrix = _calculate_steady_incidence_matrix(equations, qids)
    ```
    ................................................................................
    """
    qid_to_column = { qid: column for column, qid in enumerate(wrt_any_qids) }
    def token_within_quantities(tok: Token, /, ) -> bool:
        r"""
    ................................................................................
    ==Function: token_within_quantities==

    Checks whether a given token belongs to a specified set of quantities. Useful for 
    validating variable usage in model equations.

    ### Input arguments ###
    ???+ input "token: _nq.Token"
        The token to check.
    ???+ input "quantities: tuple[Any, ...]"
        A tuple of quantities to search within.

    ### Returns ###
    ???+ returns "bool"
        `True` if the token is within the quantities, `False` otherwise.

    ### Example ###
    ```python
        is_valid = token_within_quantities(token, quantities)
    ```
    ................................................................................
    """
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
    /,
) -> bool:
    r"""
    ................................................................................
    ==Function: _resolve_split_into_blocks==

    Determines whether to split the steady-state equations into blocks for solving. 
    Automatically resolves based on user input and the presence of a steady plan.

    ### Input arguments ###
    ???+ input "split_into_blocks: bool | None"
        User-specified preference for splitting into blocks. If `None`, automatic 
        resolution is used.
    ???+ input "plan: SteadyPlan | None"
        A steady-state plan. If provided, it affects the splitting decision.

    ### Returns ###
    ???+ returns "bool"
        `True` if the equations should be split into blocks, `False` otherwise.

    ### Example ###
    ```python
        should_split = _resolve_split_into_blocks(None, steady_plan)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Function: _update_variant_with_final_guess==

    Updates a model variant with the results of a steady-state calculation. Applies 
    logarithmic transformations where necessary and updates variable values.

    ### Input arguments ###
    ???+ input "variant: Variant"
        The model variant to update.
    ???+ input "evaluator: Any"
        The evaluator used for the steady-state calculation.
    ???+ input "qid_to_kind: dict[int, _quantities.QuantityKind]"
        Mapping from quantity IDs to their kinds.
    ???+ input "qid_to_name: dict[int, str]"
        Mapping from quantity IDs to their names.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _update_variant_with_final_guess(variant, evaluator, qid_to_kind, qid_to_name)
    ```
    ................................................................................
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
        if qid_to_kind[qid] not in LOGGABLE_VARIABLE:
            continue
        changes_to_update.append(change)
        wrt_change_qids_to_update.append(qid)
    variant.update_changes_from_array(changes_to_update, wrt_change_qids_to_update, )
    #]


def _throw_block_error(human_block, custom_header: str, ) -> NoReturn:
    r"""
    ................................................................................
    ==Function: _throw_block_error==

    Raises an error when a steady-state block calculation fails. Provides details 
    about the failing block for debugging.

    ### Input arguments ###
    ???+ input "human_block: Any"
        Human-readable details about the block.
    ???+ input "custom_header: str"
        Custom header describing the block.

    ### Returns ###
    (No return value; raises an exception)

    ### Example ###
    ```python
        _throw_block_error(human_block, "[Variant 1][Block 2]")
    ```
    ................................................................................
    """
    #[
    message = (f"Steady state calculations failed to converge in {custom_header}", )
    message += human_block.equations
    raise _wrongdoings.IrisPieError(message, )
    #]


_DEFAULT_ROOT_SETTINGS = {
    "method": "lm",
    "tol": 1e-12,
}

