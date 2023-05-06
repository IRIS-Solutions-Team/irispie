"""
"""


#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )

from . import (accessories as ea_, )
from .. import (equations as eq_, )
#]


class PlainEvaluator(ea_.EvaluatorMixin):
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


