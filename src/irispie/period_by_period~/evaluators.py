"""
Evaluators for dynamic period-by-period systems
"""


#[
from __future__ import annotations

import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities

from . import printers as _printers
from . import equators as _equators
from . import jacobians as _jacobians

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Callable, )
    from collections.abc import (Iterable, )
#]


class PeriodEvaluator:
    """
    """
    #[

    def __init__(
        self,
        wrt_qids: list[int],
        wrt_equations: Iterable[_equations.Equation],
        all_quantities: Iterable[_quantities.Quantity],
        *,
        terminal_simulator: Callable | None = None,
        context: dict | None = None,
        iter_printer_settings: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        self._wrt_qids = list(wrt_qids, )
        self.terminal_simulator = terminal_simulator
        qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
        qid_to_name = _quantities.create_qid_to_name(all_quantities, )
        #
        # Index of loglies within wrt_qids; needs to be list not tuple
        # because of numpy indexing
        self._index_logly = \
            list(_quantities.generate_where_logly(self._wrt_qids, qid_to_logly, ))
        #
        # Set up components
        self._equator = _equators.PeriodEquator(
            wrt_equations,
            context=context,
        )
        self._jacobian = _jacobians.PeriodJacobian(
            wrt_equations,
            self._wrt_qids,
            qid_to_logly,
            context=context,
        )
        self.iter_printer = _printers.PeriodIterPrinter(
            wrt_equations,
            self._wrt_qids,
            qid_to_logly,
            qid_to_name,
            **(iter_printer_settings or {}),
        )

    def get_init_guess(
        self,
        data_array: _np.ndarray,
        column: int,
    ) -> _np.ndarray:
        """
        """
        maybelog_guess = data_array[self._wrt_qids, column]
        maybelog_guess[self._index_logly] = _np.log(maybelog_guess[self._index_logly])
        return maybelog_guess

    def update(
        self,
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
        column: int,
    ) -> None:
        """
        """
        #
        # Create a copy of maybelog_guess to avoid modifying the original
        # array in scipy.optimize.root
        maybelog_guess = _np.copy(maybelog_guess, )
        maybelog_guess[self._index_logly] = \
            _np.exp(maybelog_guess[self._index_logly])
        data_array[self._wrt_qids, column] = maybelog_guess

    def eval(
        self,
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
        column: int,
    ) -> _np.ndarray:
        """
        """
        self.update(maybelog_guess, data_array, column, )
        if self.terminal_simulator:
            self.terminal_simulator(data_array, column, )
        equator = self._equator.eval(data_array, column, )
        jacobian = self._jacobian.eval(data_array, column, )
        self.iter_printer.next(maybelog_guess, equator, jacobian_calculated=True, )
        return equator, jacobian

    #]

