"""m = 
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
from .aldi import (adaptations as aa_, )
from .jacobians import (descriptors as jd_, )
#]


__all__ = [
    "SteadyEvaluator", "PlainEvaluator"
]


class _EvaluatorMixin:
    """
    """
    #[
    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self, /, ) -> int:
        """
        """
        return len(self._equations)

    def _create_evaluator_function(
        self,
        /,
        function_context: dict | None = None,
    ) -> NoReturn:
        """
        """
        function_context = aa_.add_function_adaptations_to_custom_functions(function_context)
        function_context["_array"] = np_.array
        self._xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = " , ".join(self._xtrings)
        self._func = eval(eq_.EVALUATOR_PREAMBLE + f"_array([{func_string}], dtype=float)", function_context)

    def _populate_min_max_shifts(self) -> NoReturn:
        """
        """
        self.min_shift = eq_.get_min_shift_from_equations(self._equations)
        self.max_shift = eq_.get_max_shift_from_equations(self._equations)
    #]


class SteadyEvaluator(_EvaluatorMixin):
    """
    """
    #[
    __slots__ = (
        "_t_zero", "_equations", "_quantities", "_eids", "_xtrings", "_func",
        "_incidence_matrix", "_x", "_z0", "_steady_array_updater",
        "_jacobian_descriptor",
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
        current = current if current is not None else self._z0.reshape(-1, 1, )
        x = self._steady_array_updater(self._x, current, )
        return self._func(x, self._t_zero, None, )

    def eval_with_jacobian(
        self,
        current: np_.ndarray | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        current = current if current is not None else self._z0.reshape(-1, 1, )
        x = self._steady_array_updater(self._x, current, )
        return (
            self._func(x, self._t_zero, None, ),
            self._jacobian_descriptor.eval(x, None, ),
        )

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


class PlainEvaluator(_EvaluatorMixin):
    """
    """
    #[
    __slots__ = (
        "_equations", "min_shift", "max_shift", "_func",
    )

    def __init__(
        self,
        equations: eq_.Equations,
        function_context: dir | None = None,
        /,
    ) -> NoReturn:
        self._equations = list(equations, )
        self._create_evaluator_function(function_context, )
        self._populate_min_max_shifts()

    @property
    def min_num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    def eval(
        self,
        data_array: np_.ndarray,
        columns: int | Iterable[int],
        steady_array: np_.ndarray,
        /,
    ) -> np_.ndarray:
        """
        """
        return self._func(data_array, columns, steady_array, ).reshape(self.num_equations, -1)
    #]


