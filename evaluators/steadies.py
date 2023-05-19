"""
"""


#[
from __future__ import annotations

import enum as en_
import numpy as np_
import scipy as sp_
import copy as co_
from typing import (Self, NoReturn, Callable, )
from collections.abc import (Iterable, )

from .. import (quantities as qu_, equations as eq_, )
from ..aldi import (adaptations as aa_, )
from ..jacobians import (descriptors as jd_, )
from . import (accessories as ea_, )
#]


class SteadyEvaluator(ea_.EvaluatorMixin):
    """
    """
    #[
    __slots__ = (
        "_t_zero", "_equations", "_quantities", "_eids", "_xtrings", "_func",
        "_incidence_matrix", "_x", "_z0", "_steady_array_updater",
        "_jacobian_descriptor",
        "_x_store",
    )
    @property
    def is_jacobian_sparse(self, /, ) -> bool:
        """
        True if Jacobian is sparse, False otherwise
        """
        return (
            self._jacobian_descriptor.is_sparse
            if self._jacobian_descriptor else False
        )

    def __init__(
        self,
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        t_zero: int, 
        steady_array: np_.ndarray,
        z0: np_.ndarray,
        updater: Callable,
        jacobian_descriptor: Callable | None,
        function_context: dir | None,
        /,
        print_iter: bool | Number = True,
        **kwargs,
    ) -> NoReturn:
        """ """
        self._t_zero = t_zero
        self._equations = list(equations)
        self._quantities = list(quantities)
        self._eids = list(eq_.generate_all_eids(self._equations))
        self._create_evaluator_function(function_context)
        self._create_incidence_matrix()
        self._x = steady_array
        self._z0 = z0.reshape(-1,) if z0 is not None else None
        self._steady_array_updater = updater
        self._jacobian_descriptor = jacobian_descriptor
        self._populate_min_max_shifts()
        self._iter_printer = (
            ea_.IterPrinter(self._equations, self._quantities, every=int(print_iter), )
            if print_iter else None
        )
        self._x_store = []

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
    def num_equations(self, /, ) -> int:
        return len(self._equations, )

    @property
    def num_quantities(self, /, ) -> int:
        return len(self._quantities, )

    def update(
        self,
        current: np_.ndarray | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        current = current if current is not None else self._z0.reshape(-1, 1, )
        return self._steady_array_updater(self._x, current, )

    def eval(
        self,
        current: np_.ndarray | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        #current = current[:-1] if current is not None else self._z0.reshape(-1, 1, )
        current = current if current is not None else self._z0.reshape(-1, 1, )
        self._steady_array_updater(self._x, current, )
        f = self._func(self._x, self._t_zero, None, )
        j_done = False
        if self._iter_printer:
            self._iter_printer.next(current, f, j_done, )
        return f

    def eval_sum_of_squares(
        self,
        /,
        *args,
    ) -> tuple[float, np_.ndarray]:
        """
        """
        f, j = self.eval_with_jacobian(*args)
        sum_of_squares = np_.sum(f ** 2)
        j_sum_of_squares = 2 * f.reshape(-1, 1) * j
        j_sum_of_squares = np_.sum(j_sum_of_squares, axis=0)
        return sum_of_squares, j_sum_of_squares

    def eval_with_jacobian(
        self,
        current: np_.ndarray | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        #current = current[:-1] if current is not None else self._z0.reshape(-1, 1, )
        current = current if current is not None else self._z0.reshape(-1, 1, )
        self._steady_array_updater(self._x, current, )
        f = self._func(self._x, self._t_zero, None, )
        j = self._jacobian_descriptor.eval(self._x, None, )
        j_done = True
        if self._iter_printer:
            self._iter_printer.next(current, f, j_done, )
        return f, j

    def reset(self, /, ) -> NoReturn:
        self._iter_printer.reset() if self._iter_printer else None

    def eval_jacobian(
        self,
        current: np_.ndarray | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        current = current if current is not None else self._z0.reshape(-1, 1)
        x = self._steady_array_updater(self._x, current)
        return self._jacobian_descriptor.eval(x, None, )

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


