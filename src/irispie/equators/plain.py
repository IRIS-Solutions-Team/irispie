"""
"""


#[
import numpy as _np
from typing import (Callable, )
from collections.abc import (Iterable, )

from .. import (equations as eq_, )
from . import (abc as _abc, )
#]


class PlainEquator(_abc.EquatorMixin, ):
    """
    """
    #[
    def __init__(
        self,
        equations: eq_.Equations,
        /,
        *,
        custom_functions: dict[str, Callable] | None = None,
    ) -> None:
        self._equations = tuple(equations, )
        self._create_equator_function(custom_functions, )
        self._populate_min_max_shifts()

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
        shape = (self.num_equations, -1)
        return self._func(data_array, columns, steady_array, ).reshape(shape)
    #]


