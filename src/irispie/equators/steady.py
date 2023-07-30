"""
"""


#[
import numpy as _np
from typing import (Callable, )
from collections.abc import (Iterable, )

from .. import (equations as _eq, )
from . import (abc as _abc, )
from ..evaluators import (steady as _es, )
#]


class SteadyEquator(_abc.EquatorMixin, ):
    """
    """
    #[
    def __init__(
        self,
        equations: _eq.Equations,
        t_zero: int,
        /,
        *,
        custom_functions: dict[str, Callable] | None = None,
    ) -> None:
        self._equations = tuple(equations, )
        self._create_equator_function(custom_functions, )
        self._populate_min_max_shifts()
        self._t_zero = t_zero

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        ...
    #]


class FlatSteadyEquator(SteadyEquator, ):
    """
    """
    #[
    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        return self._func(steady_array, self._t_zero, steady_array, )
    #]


class NonflatSteadyEquator(SteadyEquator, ):
    """
    """
    #[
    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        k = _es.NONFLAT_SHIFT
        return _np.hstack((
            self._func(steady_array, self._t_zero, steady_array, ),
            self._func(steady_array, self._t_zero+k, steady_array, ),
        ))
    #]


