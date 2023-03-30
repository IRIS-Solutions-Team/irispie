"""
"""

#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, Callable, )
import numpy as np_

from . import (variants as va_, )
from .. import (equations as eq_, )
from .. import (quantities as qu_, )
from .. import (evaluators as ev_, )
#]


class SteadyEvaluatorMixin:
    """
    """
    #[
    def _create_steady_evaluator(
        self,
        variant: va_.Variant,
        equations: eq_.Equations,
        quantities : qu_.Quantities,
        /,
    ) -> Callable:
        """
        """
        equations = list(equations, )
        quantities = list(quantities, )
        #
        shift_vec, t_zero = _prepare_time_shifts(equations, )
        qid_to_logly = self.create_qid_to_logly()
        steady_array = variant.create_steady_array(qid_to_logly, num_columns=shift_vec.shape[1], shift_in_first_column=-t_zero, )
        maybelog_levels, maybelog_changes, qids, index_logly = _prepare_maybelog_initial_guesses(quantities, variant, )
        maybelog_guess = maybelog_levels
        #
        def steady_array_updater(
            _steady_array: np_.ndarray,
            _maybelog_guess: np_.ndarray,
            /,
        ) -> np_.ndarray:
            maybelog_levels = maybelog_guess
            update = maybelog_levels.reshape(-1, 1) + shift_vec * maybelog_changes.reshape(-1, 1)
            update[index_logly, :] = np_.exp(update[index_logly, :])
            _steady_array[qids, :] = update
            return _steady_array
        #
        return ev_.SteadyEvaluator(
            equations,
            quantities,
            t_zero,
            steady_array,
            maybelog_guess,
            steady_array_updater,
        )
    #]


def _prepare_maybelog_initial_guesses(
    quantities,
    variant,
    /,
) -> tuple[np_.ndarray, np_.ndarray, list[int], list[int]]:
    """
    """
    #[
    qids = list(qu_.generate_all_qids(quantities, ))
    index_logly = [ i for i, qty in enumerate(quantities, ) if qty.logly ]
    maybelog_levels = variant.levels[qids]
    maybelog_changes = variant.changes[qids]
    maybelog_levels[index_logly] = np_.log(maybelog_levels[index_logly])
    maybelog_changes[index_logly] = np_.log(maybelog_changes[index_logly])
    maybelog_changes[np_.isnan(maybelog_changes)] = 0
    return maybelog_levels, maybelog_changes, qids, index_logly
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

