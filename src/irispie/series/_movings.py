"""
Time series mixin for moving sum, average, product
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Callable, )
import numpy as _np
import functools as _ft
#]


class MovingMixin:
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
        new = self.copy()
        window = window \
            if window is not None \
            else new._get_default_moving_window()
        window_length = -window
        data = _np.pad(
            new.data,
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
        new._replace_data(new_data, )
        return new

    def _get_default_moving_window(self, /, ) -> int:
        """
        Derive default moving window from time series frequency
        """
        return -self.frequency.value \
            if self.frequency is not None and self.frequency.value > 0 \
            else -4

    #]

