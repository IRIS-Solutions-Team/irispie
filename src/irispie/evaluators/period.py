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
from ..simultaneous import variants as _variants

from . import printers as _printers
#]


class PeriodEvaluator:
    """
    """
    #[

    def __init__(
        self,
        variant: _variants.Variant,
        all_quantities: Iterable[_quantities.Quantity],
        wrt_equations: Iterable[_equations.Equation],
        wrt_qids: list[int],
        *,
        context: dict | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        wrt_equations = tuple(wrt_equations, )
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
        self._equator = _equators.Period(
            wrt_equations,
            t_zero,
            context=context,
        )
        self._jacobian = self._jacobian_factory(
            wrt_equations,
            self.wrt_qids,
            qid_to_logly,
            context=context,
            first_column_to_eval = None,
            num_columns_to_eval = 1,
        )
        iter_printer_settings = iter_printer_settings or {}
        self.iter_printer = self._iter_printer_factory(
            wrt_equations,
            self.wrt_qids,
            qid_to_logly,
            qid_to_name,
            **iter_printer_settings,
        )

    #]

