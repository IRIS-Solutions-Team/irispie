"""
Steady equator
"""


#[
from __future__ import annotations

import numpy as _np
from typing import (Callable, Protocol, )
from collections.abc import (Iterable, )

from .. import equations as _equations
from ..equators import plain as _plain
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
        /,
        *,
        context: dict[str, Callable] | None = None,
    ) -> None:
        self._equator = _plain.PlainEquator(equations, context=context, )

    def eval(
        self,
        steady_array: _np.ndarray,
        column_offset: int,
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
        column_offset: int,
        /,
    ) -> _np.ndarray:
        """
        """
        return self._equator.eval(steady_array, column_offset, )

    #]


class NonflatSteadyEquator(SteadyEquator, ):
    """
    """
    #[

    # Assigned in the evaluator
    NONFLAT_STEADY_SHIFT = ...

    def eval(
        self,
        steady_array: _np.ndarray,
        column_offset: int,
        /,
    ) -> _np.ndarray:
        """
        """
        return _np.hstack((
            self._equator.eval(steady_array, column_offset, ),
            self._equator.eval(steady_array, column_offset + self.NONFLAT_STEADY_SHIFT, ),
        ))
    #]


