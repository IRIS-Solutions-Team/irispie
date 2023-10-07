"""
Steady state evaluator
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from types import (EllipsisType, )
from typing import (Any, )
from numbers import (Number, )
import itertools as _it
import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities
from ..equators import steady as _equators
from ..jacobians import steady as _jacobians
from ..models import variants as _variants

from . import printers as _printers
#]


_NONFLAT_STEADY_SHIFT = 1
_NonflatSteadyEquator = _equators.NonflatSteadyEquator
_NonflatSteadyJacobian = _jacobians.NonflatSteadyJacobian
_NonflatSteadyEquator.NONFLAT_STEADY_SHIFT = _NONFLAT_STEADY_SHIFT
_NonflatSteadyJacobian.NONFLAT_STEADY_SHIFT = _NONFLAT_STEADY_SHIFT


class SteadyEvaluator:
    """
    """
    #[
    _equator_factory: _equators.SteadyEquatorProtocol | EllipsisType = ...
    _jacobian_factory: _jacobians.SteadyJacobianProtocol | EllipsisType = ...
    _iter_printer_factory: _printers.IterPrinter | EllipsisType = ...

    def __init__(
        self,
        variant: _variants.Variant,
        all_quantities: Iterable[_quantities.Quantity],
        wrt_equations: Iterable[_equations.Equation],
        wrt_qids_levels: list[int],
        wrt_qids_changes: list[int],
        /,
        custom_functions: dict | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        wrt_equations = list(wrt_equations, )
        #
        # Create an overall long list `self.wrt_qids` comprising qids from both
        # levels and changes, and then logical indices
        # `self._bool_index_wrt_levels` and `._bool_index_wrt_changes` for
        # accessing the levels and changes within the long list
        #
        self._merge_levels_and_changes(wrt_qids_levels, wrt_qids_changes, )
        #
        qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
        qid_to_name = _quantities.create_qid_to_name(all_quantities, )
        #
        shift_vec, t_zero = _prepare_time_shifts(wrt_equations, )
        self._shift_vec = shift_vec
        self._num_columns = shift_vec.shape[1]
        #
        self._index_logly = list(_quantities.generate_index_logly(self.wrt_qids, qid_to_logly, ))
        #
        # Vectors self._maybelog_init_levels and
        # self._maybelog_init_changes are the initial guesses from the
        # variant data with missing values filled in for the long list of
        # wrt_qids. The actual values iterated over are
        # self.maybelog_levels[self._bool_index_wrt_levels] and
        # self.maybelog_changes[self._bool_index_wrt_changes].
        #
        maybelog_levels, maybelog_changes = variant.retrieve_maybelog_values_for_qids(self.wrt_qids, qid_to_logly, )
        maybelog_levels, maybelog_changes = _fill_missing(maybelog_levels, maybelog_changes, )
        self._maybelog_init_levels = maybelog_levels
        self._maybelog_init_changes = maybelog_changes
        #
        self._num_levels = sum(self._bool_index_wrt_levels)
        self._num_changes = sum(self._bool_index_wrt_changes)
        #
        self.init_guess = _np.hstack((
            self._maybelog_init_levels[self._bool_index_wrt_levels],
            self._maybelog_init_changes[self._bool_index_wrt_changes],
        ))
        #
        self._steady_array = variant.create_steady_array(qid_to_logly, num_columns=shift_vec.shape[1], shift_in_first_column=-t_zero, )
        self._update_steady_array(self.init_guess)
        #
        # Set up components
        self._equator = self._equator_factory(
            wrt_equations,
            t_zero,
            custom_functions=custom_functions,
        )
        self._jacobian = self._jacobian_factory(
            wrt_equations,
            self.wrt_qids,
            qid_to_logly,
            custom_functions=custom_functions,
        )
        iter_printer_settings = iter_printer_settings or {}
        self.iter_printer = self._iter_printer_factory(
            wrt_equations,
            self.wrt_qids,
            qid_to_logly,
            qid_to_name,
            **iter_printer_settings,
        )

    def eval(
        self,
        maybelog_guess: _np.ndarray,
        /,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        """
        self._update_steady_array(maybelog_guess, )
        equator = self._equator.eval(self._steady_array, )
        jacobian = self._jacobian.eval(self._steady_array, )
        jacobian = jacobian[:, self._bool_index_wrt_levels + self._bool_index_wrt_changes]
        self.iter_printer.next(maybelog_guess, equator, True, )
        return equator, jacobian

    def _merge_levels_and_changes(
        self,
        wrt_qids_levels: list[str],
        wrt_qids_changes: list[str],
        /,
    ) -> None:
        """
        """
        ...

    def _update_steady_array(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> None:
        """
        """
        ...

    def extract_levels(
        self,
        guess: _np.ndarray,
        /,
    ) -> tuple[_np.ndarray, tuple[int, ...]]:
        """
        """
        levels = self._get_maybelog_levels(guess, )
        levels[self._index_logly] = _np.exp(levels[self._index_logly])
        levels = levels[self._bool_index_wrt_levels]
        wrt_qids_levels = tuple(_it.compress(self.wrt_qids, self._bool_index_wrt_levels, ))
        return levels, wrt_qids_levels

    def extract_changes(
        self,
        guess: _np.ndarray,
        /,
    ) -> tuple[_np.ndarray, tuple[int, ...]]:
        """
        """
        changes = self._get_maybelog_changes(guess, )
        changes[self._index_logly] = _np.exp(changes[self._index_logly])
        changes = changes[self._bool_index_wrt_changes]
        wrt_qids_changes = tuple(_it.compress(self.wrt_qids, self._bool_index_wrt_changes, ))
        return changes, wrt_qids_changes
    #]


class FlatSteadyEvaluator(SteadyEvaluator, ):
    """
    """
    #[
    _equator_factory = _equators.FlatSteadyEquator
    _jacobian_factory = _jacobians.FlatSteadyJacobian
    _iter_printer_factory = _printers.FlatSteadyIterPrinter

    def __init__(
        self,
        variant: _variants.Variant,
        all_quantities: Iterable[_quantities.Quantity],
        *args,
        **kwargs,
    ) -> None:
        """
        """
        super().__init__(variant, all_quantities, *args, **kwargs, )
        qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
        variant.zero_changes(qid_to_logly, )

    def _merge_levels_and_changes(
        self,
        wrt_qids_levels: list[int],
        wrt_qids_changes: list[int],
        /,
    ) -> None:
        self.wrt_qids = list(set(wrt_qids_levels))
        self._bool_index_wrt_levels = [qid in wrt_qids_levels for qid in self.wrt_qids]
        self._bool_index_wrt_changes = []

    def _get_maybelog_levels(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        new_maybelog_levels = _np.copy(self._maybelog_init_levels)
        if self._bool_index_wrt_levels:
            new_maybelog_levels[self._bool_index_wrt_levels] = current_guess
        return new_maybelog_levels

    def _get_maybelog_changes(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        return _np.zeros(self._maybelog_init_changes.shape, dtype=float, )

    def _update_steady_array(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> None:
        """
        """
        new_maybelog_levels = self._get_maybelog_levels(current_guess, )
        new_paths = _np.repeat(new_maybelog_levels.reshape(-1, 1), self._num_columns, axis=1, )
        new_paths[self._index_logly, :] = _np.exp(new_paths[self._index_logly, :])
        self._steady_array[self.wrt_qids, :] = new_paths
    #]


class NonflatSteadyEvaluator(SteadyEvaluator, ):
    """
    """
    #[
    _equator_factory = _NonflatSteadyEquator
    _jacobian_factory = _NonflatSteadyJacobian
    _iter_printer_factory = _printers.NonflatSteadyIterPrinter

    def _merge_levels_and_changes(
        self,
        wrt_qids_levels: list[str],
        wrt_qids_changes: list[str],
        /,
    ) -> None:
        self.wrt_qids = list(set(wrt_qids_levels) | set(wrt_qids_changes))
        self._bool_index_wrt_levels = [qid in wrt_qids_levels for qid in self.wrt_qids]
        self._bool_index_wrt_changes = [qid in wrt_qids_changes for qid in self.wrt_qids]

    def _get_maybelog_levels(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        new_maybelog_levels = _np.copy(self._maybelog_init_levels)
        if self._bool_index_wrt_levels:
            new_maybelog_levels[self._bool_index_wrt_levels] = current_guess[:self._num_levels]
        return new_maybelog_levels

    def _get_maybelog_changes(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        new_maybelog_changes = _np.copy(self._maybelog_init_changes)
        if self._bool_index_wrt_changes:
            new_maybelog_changes[self._bool_index_wrt_changes] = current_guess[self._num_levels:]
        return new_maybelog_changes

    def _update_steady_array(
        self,
        current_guess: _np.ndarray,
        /,
    ) -> None:
        """
        """
        new_maybelog_levels = self._get_maybelog_levels(current_guess, )
        new_maybelog_changes = self._get_maybelog_changes(current_guess, )
        new_paths = new_maybelog_levels.reshape(-1, 1) + self._shift_vec * new_maybelog_changes.reshape(-1, 1)
        new_paths[self._index_logly, :] = _np.exp(new_paths[self._index_logly, :])
        self._steady_array[self.wrt_qids, :] = new_paths
    #]


def _fill_missing(
    maybelog_levels: _np.ndarray,
    maybelog_changes: _np.ndarray,
    /,
    default_level: Number = 1/9,
    default_change: Number = 0,
) -> tuple[_np.ndarray, _np.ndarray]:
    """
    """
    #[
    maybelog_levels[_np.isnan(maybelog_levels)] = default_level
    maybelog_changes[_np.isnan(maybelog_changes)] = default_change
    return maybelog_levels, maybelog_changes
    #]


def _prepare_time_shifts(
    equations: Iterable[_equations.Equation],
    /,
) -> tuple[_np.ndarray, int]:
    """
    Return [min_shift, ..., max_shift] and the index of the column with shift=0
    """
    #[
    # Add 1 to max shift to accommodate nonflat steady equators
    add_shift = 1
    #
    min_shift = _equations.get_min_shift_from_equations(equations, )
    max_shift = _equations.get_max_shift_from_equations(equations, ) + add_shift
    num_columns = -min_shift + 1 + max_shift
    shift_in_first_column = min_shift
    shift_vec = _np.array(range(min_shift, max_shift+1, ), dtype=float, ).reshape(1, -1, )
    t_zero = -min_shift
    return shift_vec, t_zero
    #]


