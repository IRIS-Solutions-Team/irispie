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
from ._equators import (Equator, )
from ._jacobians import (Jacobian, )
from ..evaluators.base import (Evaluator, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Callable, )
    from collections.abc import (Iterable, Sequence, )
    from ..fords.terminators import (Terminator, )
#]


def _create_update_map(
    wrt_spots: Iterable[Token],
) -> ArrayMap:
    """
Create a map for updating the data array with the new guess.
    """
    update_map = ArrayMap()
    lhs_rows, lhs_columns = zip(*wrt_spots, )
    lhs_rows = _np.array(lhs_rows, dtype=int, )
    lhs_columns = _np.array(lhs_columns, dtype=int, )
    update_map.lhs = (lhs_rows, lhs_columns, )
    return update_map


def create_evaluator(
    wrt_spots: Iterable[Token],
    columns_to_eval: Sequence[int],
    wrt_equations: Iterable[_equations.Equation],
    all_quantities: Iterable[_quantities.Quantity],
    terminator: Terminator | None,
    context: dict | None,
) -> SimpleNamespace:
    """
    """
    #[
    num_columns_to_eval = len(columns_to_eval)
    wrt_spots = list(wrt_spots, )
    needs_terminal = terminator is not None
    #
    qid_to_logly = _quantities.create_qid_to_logly(all_quantities, )
    qid_to_name = _quantities.create_qid_to_name(all_quantities, )
    #
    # Index of loglies within wrt_spots; needs to be list not tuple
    # because of numpy indexing
    index_logly = list(_quantities.generate_where_logly(
        ( tok.qid for tok in wrt_spots ), qid_to_logly,
    ))

    update_map = _create_update_map(wrt_spots, )

    equator = Equator(
        wrt_equations,
        columns=columns_to_eval,
        context=context,
    )

    jacobian_wrt_spots = tuple(wrt_spots)
    if needs_terminal:
        jacobian_wrt_spots += tuple(terminator.terminal_wrt_spots)

    jacobian = Jacobian(
        wrt_equations,
        jacobian_wrt_spots,
        qid_to_logly,
        sparse=True,
        context=context,
        columns_to_eval=columns_to_eval,
        terminator=terminator,
    )

    def get_init_guess(data_array: _np.ndarray, /, ) -> _np.ndarray:
        """
        """
        maybelog_guess = data_array[update_map.lhs[0], update_map.lhs[1]]
        maybelog_guess[index_logly] = _np.log(maybelog_guess[index_logly])
        return maybelog_guess

    def update(
        maybelog_guess: _np.ndarray,
        data_array: _np.ndarray,
    ) -> None:
        """
        """
        maybelog_guess = _np.copy(maybelog_guess, )
        maybelog_guess[index_logly] = _np.exp(maybelog_guess[index_logly])
        data_array[update_map.lhs[0], update_map.lhs[1]] = maybelog_guess

    def eval_func_jacob(
        maybelog_guess: _np.ndarray | None,
        data_array: _np.ndarray,
    ) -> tuple[_np.ndarray, _sp.sparse.csc_matrix]:
        """
        """
        if maybelog_guess is not None:
            update(maybelog_guess, data_array, )
        #
        if needs_terminal:
            terminator.terminate_simulation(data_array, )
        #
        # The equator evaluates to a tuple of 1D arrays, each array for
        # one equation across all periods. Reorganize into a 1D array ordered
        # as follows:
        #
        # [eq1[t=0], eq2[t=0], ..., eq1[t=1], eq2[t=1], ...]
        #
        equator_outcome = equator.eval(data_array, )
        equator_outcome = _np.vstack(equator_outcome, ).flatten(order="Fortran"[0], )
        #
        jacobian_outcome = jacobian.eval(data_array, )
        if needs_terminal:
            jacobian_outcome = terminator.terminate_jacobian(jacobian_outcome, )
        return equator_outcome, jacobian_outcome, 

    def eval_func(
        maybelog_guess: _np.ndarray | None,
        data_array: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        if maybelog_guess is not None:
            update(maybelog_guess, data_array, )
        if needs_terminal:
            terminator.terminate_simulation(data_array, )
        equator_outcome = equator.eval(data_array, )
        return _np.vstack(equator_outcome, ).flatten(order="F", )

    def eval_jacob(
        maybelog_guess: _np.ndarray | None,
        data_array: _np.ndarray,
    ) -> _sp.sparse.csc_matrix:
        """
        """
        if maybelog_guess is not None:
            update(maybelog_guess, data_array, )
        if needs_terminal:
            terminator.terminate_simulation(data_array, )
        jacobian_outcome = jacobian.eval(data_array, )
        if needs_terminal:
            jacobian_outcome = terminator.terminate_jacobian(jacobian_outcome, )
        return jacobian_outcome

    return Evaluator(
        eval_func_jacob=eval_func_jacob,
        eval_func=eval_func,
        eval_jacob=eval_jacob,
        update=update,
        get_init_guess=get_init_guess,
    )

    #]


