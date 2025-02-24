"""
Evaluators for dynamic period-by-period systems
"""


#[

from __future__ import annotations

import numpy as _np
from collections import namedtuple
from neqs import IterPrinter

from .. import equations as _equations
from .. import quantities as _quantities
from ._equators import Equator
from ._jacobians import Jacobian

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from collections.abc import Iterable

#]


_Evaluator = namedtuple(
    "_Evaluator",
    ("evaluate", "update", "get_init_guess", "iter_printer", ),
)


def create_evaluator(
    wrt_qids: Iterable[int],
    wrt_equations: Iterable[_equations.Equation],
    all_quantities: Iterable[_quantities.Quantity],
    terminal_simulator: Callable | None,
    context: dict | None,
    iter_printer_settings: dict[str, Any] | None,
) -> tuple[Callable, Callable]:
    """
    """
    wrt_qids = list(wrt_qids, )
    qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
    qid_to_name = _quantities.create_qid_to_name(all_quantities, )
    #
    # Index of loglies within wrt_qids; needs to be list not tuple
    # because of numpy indexing
    index_logly = \
        list(_quantities.generate_where_logly(wrt_qids, qid_to_logly, ))

    equator = Equator(
        wrt_equations,
        context=context,
    )

    jacobian = Jacobian(
        wrt_equations,
        wrt_qids,
        qid_to_logly,
        context=context,
    )

    iter_printer = IterPrinter(
        equation=wrt_equations,
        qids=wrt_qids,
        qid_to_logly=qid_to_logly,
        qid_to_name=qid_to_name,
        **(iter_printer_settings or {}),
    )

    def get_init_guess(
        data_array: _np.ndarray,
        columns: int | _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        maybelog_guess = data_array[wrt_qids, columns]
        maybelog_guess[index_logly] = _np.log(maybelog_guess[index_logly])
        return maybelog_guess

    def update(
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
        columns: int | _np.ndarray,
    ) -> None:
        """
        """
        #
        # Create a copy of maybelog_guess to avoid modifying the original
        # array in scipy.optimize.root
        maybelog_guess = _np.copy(maybelog_guess, )
        maybelog_guess[index_logly] = \
            _np.exp(maybelog_guess[index_logly])
        data_array[wrt_qids, columns] = maybelog_guess

    def evaluate(
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
        columns: int | _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        update(maybelog_guess, data_array, columns, )
        if terminal_simulator:
            terminal_simulator(data_array, columns, )
        equator_outcome = equator.eval(data_array, columns, )
        jacobian_outcome = jacobian.eval(data_array, columns, )
        iter_printer.next(maybelog_guess, equator_outcome, jacobian_calculated=True, )
        return equator_outcome, jacobian_outcome,

    return _Evaluator(evaluate, update, get_init_guess, iter_printer, )

    #]

