"""
Steady equator
"""


#[
from __future__ import annotations

import numpy as _np
from typing import (Callable, Protocol, )
from collections.abc import (Iterable, )

from .. import equations as _equations
from . import plain as _plain
#]


class SteadyEquatorProtocol:
    """
    """
    #[
    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        ...
    #]


class SteadyEquator:
    """
    """
    #[
    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        t_zero: int,
        /,
        *,
        custom_functions: dict[str, Callable] | None = None,
    ) -> None:
        self._equator = _plain.PlainEquator(equations, custom_functions=custom_functions, )
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
        return self._equator.eval(steady_array, self._t_zero, steady_array, )
    #]


class NonflatSteadyEquator(SteadyEquator, ):
    """
    """
    #[
    NONFLAT_STEADY_SHIFT: int = 1

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        k = self.NONFLAT_STEADY_SHIFT
        return _np.hstack((
            self._equator.eval(steady_array, self._t_zero, steady_array, ),
            self._equator.eval(steady_array, self._t_zero+k, steady_array, ),
        ))
    #]


