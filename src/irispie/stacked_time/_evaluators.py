"""
Evaluators for dynamic period-by-period systems
"""


#[
from __future__ import annotations

import numpy as _np
from types import (SimpleNamespace, )

from .. import equations as _equations
from .. import quantities as _quantities
from ..aldi.maps import (ArrayMap, )
from ..incidences.main import (Token, )
from ._printers import (IterPrinter, )
from ._equators import (Equator, )
from ._jacobians import (Jacobian, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Callable, )
    from collections.abc import (Iterable, )
#]


def _create_update_map(
    wrt_tokens: Iterable[Token],
) -> ArrayMap:
    """
Create a map for updating the data array with the new guess.
    """
    update_map = ArrayMap()
    lhs_rows, lhs_columns = zip(*wrt_tokens, )
    lhs_rows = _np.array(lhs_rows, dtype=int, )
    lhs_columns = _np.array(lhs_columns, dtype=int, )
    update_map.lhs = (lhs_rows, lhs_columns, )
    return update_map


def create_evaluator_closure(
    wrt_tokens: Iterable[Token],
    extended_wrt_tokens: Iterable[Token],
    #
    wrt_equations: Iterable[_equations.Equation],
    all_quantities: Iterable[_quantities.Quantity],
    terminate_simulation: Callable | None,
    context: dict | None,
    iter_printer_settings: dict[str, Any] | None,
    num_columns_to_eval: int,
) -> tuple[Callable, Callable]:
    """
    """
    #[

    wrt_tokens = list(wrt_tokens, )
    extended_wrt_tokens = list(extended_wrt_tokens, )
    qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
    qid_to_name = _quantities.create_qid_to_name(all_quantities, )
    #
    # Index of loglies within wrt_tokens; needs to be list not tuple
    # because of numpy indexing
    index_logly = list(_quantities.generate_where_logly(
        ( tok.qid for tok in wrt_tokens ), qid_to_logly,
    ))

    update_map = _create_update_map(wrt_tokens, )

    equator = Equator(
        wrt_equations,
        context=context,
    )

    jacobian = Jacobian(
        wrt_equations,
        wrt_tokens + extended_wrt_tokens,
        qid_to_logly,
        context=context,
        num_columns_to_eval=num_columns_to_eval,
    )

    iter_printer = IterPrinter(
        wrt_equations,
        tuple(tok.qid for tok in wrt_tokens),
        qid_to_logly,
        qid_to_name,
        **(iter_printer_settings or {}),
    )

    def get_init_guess(
        data_array: _np.ndarray,
        columns_to_eval: int | _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        column_offset = columns_to_eval[0]
        maybelog_guess = data_array[update_map.lhs[0], update_map.lhs[1]+column_offset]
        maybelog_guess[index_logly] = _np.log(maybelog_guess[index_logly])
        return maybelog_guess

    def update(
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
        columns_to_eval: _np.ndarray,
    ) -> None:
        """
        """
        column_offset = columns_to_eval[0]
        # Create a copy of maybelog_guess to avoid modifying the original
        # array in scipy.optimize.root
        maybelog_guess = _np.copy(maybelog_guess, )
        maybelog_guess[index_logly] = _np.exp(maybelog_guess[index_logly])
        data_array[update_map.lhs[0], update_map.lhs[1]+column_offset] = maybelog_guess

    def evaluate(
        maybelog_guess: _np.ndarray | None,
        data_array: _np.ndarray,
        columns_to_eval: _np.ndarray,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        """
        first_column = columns_to_eval[0]
        last_column = columns_to_eval[-1]
        if maybelog_guess is not None:
            update(maybelog_guess, data_array, columns_to_eval, )
        if terminate_simulation is not None:
            terminate_simulation(data_array, last_simulation=last_column, )
        #
        # The equator evaluates to a tuple of 1D arrays, each tuple element for
        # one equation across all periods. Reorganize into a 1D array ordered
        # as follows:
        #
        # [eq1[t=0], eq2[t=0], ..., eq1[t=1], eq2[t=1], ...]
        #
        equator_outcome = equator.eval(data_array, columns_to_eval, )
        equator_outcome = _np.vstack(equator_outcome, ).flatten(order="Fortran"[0], )
        #
        jacobian_outcome = jacobian.eval(data_array, column_offset=first_column, )
        #
        if maybelog_guess is not None:
            iter_printer.next(maybelog_guess, equator_outcome, jacobian_calculated=True, )
        return equator_outcome, jacobian_outcome,

    return SimpleNamespace(
        evaluate=evaluate,
        update=update,
        get_init_guess=get_init_guess,
    )

    #]

