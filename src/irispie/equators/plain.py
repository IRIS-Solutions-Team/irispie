"""
Plain equator
"""


#[
from __future__ import annotations

import numpy as _np

from .. import equations as _equations
from .. import makers as _makers

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable
    from collections.abc import Iterable
#]


EQUATOR_ARGS = ("x", "t", )


class PlainEquator:
    """
    """
    #[

    _state_slots = (
        "_context",
        "_equations",
        "_func_str",
        "_columns",
        "min_shift",
        "max_shift",
    )

    _nonstate_slots = (
        "_func",
    )

    __slots__ = _state_slots + _nonstate_slots

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        columns: Iterable[int] | None = None,
        context: dict[str, Callable] | None = None,
    ) -> None:
        """
        """
        self._equations = tuple(equations)
        self._populate_min_max_shifts()
        self._populate_columns(columns, )
        self._context = context
        self._create_function()

    @property
    def humans(self, /, ) -> tuple[str]:
        return tuple(e.human for e in self._equations)

    @property
    def num_equations(self, /, ) -> int:
        """
        """
        return len(self._equations)

    def _create_function(self, /, ) -> None:
        """
        """
        joined_xtrings = "  ,  ".join(i.xtring for i in self._equations)
        expression = "(" + joined_xtrings + " , )"
        self._func, self._func_str, *_ \
            = _makers.make_function("__equator", EQUATOR_ARGS, expression, self._context, )

    def _populate_min_max_shifts(self, /, ) -> None:
        """
        """
        self.min_shift = _equations.get_min_shift_from_equations(self._equations, )
        self.max_shift = _equations.get_max_shift_from_equations(self._equations, )

    def _populate_columns(self, columns: Iterable[int] | None, /, ) -> None:
        """
        """
        if columns is None:
            self._columns = None
        elif isinstance(columns, int):
            self._columns = columns
        else:
            self._columns = _np.array(columns, dtype=int, )

    @property
    def min_num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    def eval(
        self,
        data_array: _np.ndarray,
        columns: int | _np.ndarray | None = None,
    ) -> _np.ndarray:
        """
        """
        if columns is None:
            columns = self._columns
        return self._func(data_array, columns, )

    def eval_as_array(
        self,
        *args,
        **kwargs,
    ) -> _np.ndarray:
        """
        """
        value = self.eval(*args, **kwargs, )
        return _np.array(value, dtype=_np.float64, ).reshape(self.num_equations, -1, )

    def __getstate__(self, ) -> dict[str, Any]:
        """
        """
        return {
            k: getattr(self, k)
            for k in self._state_slots
        }

    def __setstate__(self, state: dict[str, Any], ) -> None:
        """
        """
        for n in self._state_slots:
            setattr(self, n, state[n], )
        self._create_function()

    #]

