"""
"""

#[
from __future__ import annotations
# from IPython import embed

import enum as en_
import numpy as np_
import copy as co_
from typing import (Self, NoReturn, Callable, )
from collections.abc import (Iterable, )

from . import (quantities as qu_, )
from . import (equations as eq_, )
from .functions import *
#]


__all__ = [
    "SteadyEvaluator", "PlainEvaluator"
]


class SteadyEvaluator:
    """
    """
    #[
    def __init__(
        self,
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        t_zero: int, 
        steady_array: np_.ndarray,
        z0: np_.ndarray,
        updater: Callable,
        /,
    ) -> NoReturn:
        self._t_zero = t_zero
        self._equations = list(equations)
        self._quantities = list(quantities)
        self._eids = list(eq_.generate_all_eids(self._equations))
        self._create_evaluator_function()
        self._create_incidence_matrix()
        self._x = steady_array
        self._z0 = z0.reshape(-1,) if z0 is not None else None
        self._steady_array_updater = updater
        self.min_shift = eq_.get_min_shift_from_equations(self._equations)
        self.max_shift = eq_.get_max_shift_from_equations(self._equations)

    @property
    def initial_guess(self, /, ) -> np_.ndarray:
        return np_.copy(self._z0)

    @property
    def steady_array(self, /, ) -> np_.ndarray:
        return np_.copy(self._x)

    @property
    def quantities_human(self, /, ) -> Iterable[str]:
        return [ qty.human for qty in self._quantities ]

    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self, /, ) -> int:
        return len(self._equations)

    @property
    def num_quantities(self, /, ) -> int:
        return len(self._quantities)

    def eval(self, current: np_.ndarray | None = None, /, ):
        """
        """
        current = current if current is not None else self._z0.reshape(-1, 1)
        x = self._steady_array_updater(self._x, current)
        return self._func(x, self._t_zero, x)

    def _create_evaluator_function(self, /, ) -> NoReturn:
        """
        """
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t, L: np_.array([{func_string}], dtype=float)")

    def _create_incidence_matrix(self, /, ) -> NoReturn:
        """
        """
        matrix = np_.zeros((self.num_equations, self.num_quantities), dtype=bool)
        qids = list(qu_.generate_all_qids(self._quantities))
        qid_to_column = { qid: column for column, qid in enumerate(qids) }
        for row_index, eqn in enumerate(self._equations):
            column_indices = list(set(
                qid_to_column[tok.qid]
                for tok in eqn.incidence if tok.qid in qids
            ))
            matrix[row_index, column_indices] = True
        self.incidence_matrix = matrix
#]


class PlainEvaluator:
    """
    """
    #[
    __slots__ = [
        "_equations", "min_shift", "max_shift", "_func"
    ]

    def __init__(
        self,
        equations: eq_.Equations,
        /,
    ) -> NoReturn:
        self._equations = list(equations)
        self.min_shift = eq_.get_min_shift_from_equations(self._equations)
        self.max_shift = eq_.get_max_shift_from_equations(self._equations)
        self._create_evaluator_function()

    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def min_num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    @property
    def num_equations(self, /, ) -> int:
        return len(self._equations)

    def eval(self, data_array: np_.ndarray, columns, steady_array, /, ) -> np_.ndarray:
        """
        """
        return self._func(data_array, columns, steady_array, ).reshape(self.num_equations, -1)

    def _create_evaluator_function(self, /, ) -> NoReturn:
        """
        """
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t, L: np_.array([{func_string}], dtype=float)")
#]


