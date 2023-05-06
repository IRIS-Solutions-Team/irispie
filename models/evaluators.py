"""
"""

#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )
import numpy as np_
import functools as ft_

from .. import (equations as eq_, quantities as qu_, wrongdoings as wd_, )
from ..jacobians import (descriptors as jd_, )
from ..evaluators import (steadies as es_, )

from . import (variants as va_, )
#]


"""
Equation kinds that are allowed in SteadyEvaluator objects
"""
STEADY_EVALUATOR_EQUATION = (
    eq_.EquationKind.TRANSITION_EQUATION
    | eq_.EquationKind.MEASUREMENT_EQUATION
)


"""
Quantity kinds that are allowed in SteadyEvaluator objects
"""
STEADY_EVALUATOR_QUANTITY = (
    qu_.QuantityKind.TRANSITION_VARIABLE
    | qu_.QuantityKind.MEASUREMENT_VARIABLE
)


class SteadyEvaluatorMixin:
    """
    """
    #[
    def create_steady_evaluator(
        self,
        /,
        equations: eq_.Equations | None = None,
        quantities: qu_.Quantities | None = None,
        **kwargs,
    ) -> es_.SteadyEvaluator:
        """
        Create a steady-state Evaluator object for the primary variant of this Model
        """
        allowed_equations = eq_.generate_equations_of_kind(self._invariant._steady_equations, STEADY_EVALUATOR_EQUATION)
        equations, invalid_equations = eq_.validate_selection_of_equations(allowed_equations, equations)
        if invalid_equations:
            raise wd_.IrisPieError(
                ["Expecting measurement and/or transition equations, getting"]
                + [ eqn.human for eqn in invalid_equations ]
            )
        #
        allowed_quantities = qu_.generate_quantities_of_kind(self._invariant._quantities, STEADY_EVALUATOR_QUANTITY)
        quantities, invalid_quantities = qu_.validate_selection_of_quantities(allowed_quantities, quantities)
        if invalid_quantities:
            raise wd_.IrisPieError(
                ["Expecting names of measurement and/or transition variables, getting"]
                + [ qty.human for qty in invalid_quantities ]
            )
        #
        return self._create_steady_evaluator(
            self._variants[0],
            equations,
            quantities,
            **kwargs,
        )

    def _create_steady_evaluator(
        self,
        variant: va_.Variant,
        equations: eq_.Equations,
        quantities : qu_.Quantities,
        /,
        **kwargs,
    ) -> Callable:
        """
        Create steady evaluator for the a given variant of this Model
        """
        equations = list(equations, )
        quantities = list(quantities, )
        function_context = self._invariant._function_context
        #
        shift_vec, t_zero = _prepare_time_shifts(equations, )
        qid_to_logly = self.create_qid_to_logly()
        steady_array = variant.create_steady_array(qid_to_logly, num_columns=shift_vec.shape[1], shift_in_first_column=-t_zero, )
        maybelog_levels, maybelog_changes, qids, index_logly = _prepare_maybelog_initial_guesses(quantities, variant, **kwargs, )
        maybelog_initial_guess = maybelog_levels
        #
        steady_array_updater = ft_.partial(
            _steady_array_updater, maybelog_changes=maybelog_changes,
            shift_vec=shift_vec, qids=qids, index_logly=index_logly,
        )
        #
        jacobian_descriptor = jd_.Descriptor.for_flat(
            equations, quantities, qid_to_logly, function_context, **kwargs,
        )
        #
        return es_.SteadyEvaluator(
            equations, quantities,
            t_zero, steady_array, maybelog_initial_guess,
            steady_array_updater, jacobian_descriptor, function_context, 
            **kwargs,
        )
    #]


def _steady_array_updater(
    steady_array: np_.ndarray,
    maybelog_initial_guess: np_.ndarray,
    /,
    maybelog_changes: np_.ndarray,
    shift_vec: np_.ndarray,
    qids: Iterable[int],
    index_logly: Iterable[int],
) -> np_.ndarray:
    """
    """
    maybelog_levels = np_.copy(maybelog_initial_guess)
    update = maybelog_levels.reshape(-1, 1) + shift_vec * maybelog_changes.reshape(-1, 1)
    update[index_logly, :] = np_.exp(update[index_logly, :])
    steady_array[qids, :] = update
    #return steady_array


def _prepare_maybelog_initial_guesses(
    quantities,
    variant,
    /,
    **kwargs,
) -> tuple[np_.ndarray, np_.ndarray, list[int], list[int]]:
    """
    """
    #[
    qids = list(qu_.generate_all_qids(quantities, ))
    index_logly = [ i for i, qty in enumerate(quantities, ) if qty.logly ]
    #
    # Extract initial guesses for levels and changes
    maybelog_levels = variant.levels[qids]
    maybelog_changes = variant.changes[qids]
    #
    # Logarithmize
    maybelog_levels[index_logly] = np_.log(maybelog_levels[index_logly])
    maybelog_changes[index_logly] = np_.log(maybelog_changes[index_logly])
    #
    # Fill missing initial guesses
    _fill_missing_maybelog_initial_guesses(maybelog_levels, maybelog_changes, **kwargs, )
    #
    return maybelog_levels, maybelog_changes, qids, index_logly
    #]


def _fill_missing_maybelog_initial_guesses(
    maybelog_levels: np_.ndarray,
    maybelog_changes: np_.ndarray,
    /,
    default_level: Number = 1/9,
    **kwargs,
) -> tuple[np_.ndarray, np_.ndarray]:
    """
    """
    #[
    maybelog_levels[np_.isnan(maybelog_levels)] = default_level
    maybelog_changes[np_.isnan(maybelog_changes)] = 0
    #]


def _prepare_time_shifts(
    equations: eq_.Equations, 
    /,
) -> tuple[np_.ndarray, int]:
    """
    """
    #[
    min_shift = eq_.get_min_shift_from_equations(equations, )
    max_shift = eq_.get_max_shift_from_equations(equations, )
    num_columns = -min_shift + 1 + max_shift
    shift_in_first_column = min_shift
    t_zero = -min_shift
    shift_vec = np_.array(range(min_shift, max_shift+1, ), dtype=float, ).reshape(1, -1, )
    return shift_vec, t_zero
    #]

