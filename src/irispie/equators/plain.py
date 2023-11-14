"""
Plain equator
"""


#[
from __future__ import annotations

import numpy as _np
from typing import (Protocol, Callable, Self, )
from collections.abc import (Iterable, )

from .. import equations as _equations
from .. import makers as _makers
#]


EQUATOR_ARGS = ("x", "t", "L", )


class PlainEquator:
    """
    """
    #[

    __slots__ = (
        "_equations",
        "_func",
        "_func_str",
        "min_shift",
        "max_shift",
    )

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        /,
        *,
        context: dict[str, Callable] | None = None,
    ) -> None:
        """
        """
        self._equations = tuple(equations)
        self._create_function(context, )
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
        context: dict | None = None,
        /,
    ) -> None:
        """
        """
        joined_xtrings = "  ,  ".join(i.xtring for i in self._equations)
        expression = "(" + joined_xtrings + " , )"
        self._func, self._func_str, *_ = \
            _makers.make_lambda(EQUATOR_ARGS, expression, context, )

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self.min_shift = _equations.get_min_shift_from_equations(self._equations, )
        self.max_shift = _equations.get_max_shift_from_equations(self._equations, )

    @property
    def min_num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    def eval(
        self,
        data_array: _np.ndarray,
        columns: int | _np.ndarray,
        steady_array: _np.ndarray | None,
        /,
    ) -> _np.ndarray:
        """
        """
        steady_array = steady_array if steady_array is not None else data_array
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

