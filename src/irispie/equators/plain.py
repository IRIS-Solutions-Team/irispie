"""
Plain equator
"""


#[
from __future__ import annotations

import numpy as _np
from typing import (Protocol, Callable, Self, )
from collections.abc import (Iterable, )

from .. import equations as _eq
from .. import quantities as _qu
from ..aldi import adaptations as _aa
#]


class PlainEquator:
    """
    """
    #[
    def __init__(
        self,
        equations: Iterable[_eq.Equation],
        /,
        *,
        custom_functions: dict[str, Callable] | None = None,
    ) -> None:
        self._equations = tuple(equations)
        self._create_function(custom_functions, )
        self._populate_min_max_shifts()

    @property
    def humans(self, /, ) -> tuple[str]:
        return tuple(e.human for e in self._equations)

    @property
    def num_equations(self, /, ) -> int:
        """
        """
        return len(self._equations)

    def _create_function(
        self,
        custom_functions: dict | None = None,
        /,
    ) -> None:
        """
        """
        custom_functions = _aa.add_function_adaptations_to_custom_functions(custom_functions, )
        joined_xtrings = "  ,  ".join(i.xtring for i in self._equations)
        self._func_str = _eq.EVALUATOR_PREAMBLE + "(" + joined_xtrings + " , )"
        self._func = eval(self._func_str, custom_functions, )

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self.min_shift = _eq.get_min_shift_from_equations(self._equations, )
        self.max_shift = _eq.get_max_shift_from_equations(self._equations, )

    @property
    def min_num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    def eval(
        self,
        data_array: _np.ndarray,
        columns: int | Iterable[int],
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        return self._func(data_array, columns, steady_array, )

    def eval_as_array(
        self,
        *args,
        **kwargs,
    ) -> _np.ndarray:
        """
        """
        value = self.eval(*args, **kwargs, )
        return _np.array(value, dtype=_np.float64, ).reshape(self.num_equations, -1, )
    #]

