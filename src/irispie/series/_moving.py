"""
Time series mixin for moving sum, average, product
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Callable, )
import numpy as _np
import functools as _ft

from .. import dates as _dates
from . import _functionalize
#]


__all__ = ()


class Inlay:
    """
    Time series mixin for moving sum, average, product
    """
    #[

    def mov_sum(self, window: int | None = None, ) -> Self:
        """
        Moving sum of time current and lagged series observations
        """
        return self._moving(_np.sum, window=window, )

    def mov_avg(self, window: int | None = None, ) -> Self:
        """
        Moving average of time current and lagged series observations
        """
        return self._moving(_np.mean, window=window, )

    def mov_mean(self, window: int | None = None, ) -> Self:
        """
        Moving average of time current and lagged series observations
        """
        return self._moving(_np.mean, window=window, )

    def mov_prod(self, window: int | None = None, ) -> Self:
        """
        Moving product of time current and lagged series observations
        """
        return self._moving(_np.prod, window=window, )

    def _moving(
        self,
        func: Callable,
        /,
        window: int | None = None,
    ) -> Self:
        """
        Backend function for moving sum, average, product
        """
        window = (
            window
            if window is not None
            else self._get_default_moving_window()
        )
        window_length = -window
        data = _np.pad(
            self.data,
            pad_width=((window_length-1, 0), (0, 0)),
            mode="constant",
            constant_values=_np.nan,
        )
        data_windows =_np.lib.stride_tricks.sliding_window_view(
            data,
            window_shape=window_length,
            axis=0,
        )
        new_data = func(data_windows, axis=2, )
        self._replace_data(new_data, )

    def _get_default_moving_window(self, /, ) -> int:
        """
        Derive default moving window from time series frequency
        """
        return (
            -self.frequency.value
            if self.frequency is not None and self.frequency.value > 0
            else -4
        )

    #]


for n in ("mov_sum", "mov_avg", "mov_mean", "mov_prod", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )

