"""
Time series mixin for moving sum, average, product
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Callable, )
import numpy as _np
import functools as _ft

from .. import pages as _pages
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
        """
        return self._moving(_np.sum, window=window, )

    def mov_avg(self, window: int | None = None, ) -> Self:
        """
        """
        return self._moving(_np.mean, window=window, )

    def mov_mean(self, window: int | None = None, ) -> Self:
        """
        """
        return self._moving(_np.mean, window=window, )

    def mov_prod(self, window: int | None = None, ) -> Self:
        """
        """
        return self._moving(_np.prod, window=window, )

    @_pages.reference(category="filtering", call_name="mov_*", )
    def _moving(
        self,
        func: Callable,
        /,
        window: int | None = None,
    ) -> Self:
        r"""
················································································

==Moving functions==


### Function for creating new Series objects ###

```
output = irispie.mov_sum(self, window=None, )
output = irispie.mov_avg(self, window=None, )
output = irispie.mov_mean(self, window=None, )
output = irispie.mov_prod(self, window=None, )
```


### Methods for changing the existing Series object in-place ###


```
self.mov_sum(window=None, )
self.mov_avg(window=None, )
self.mov_mean(window=None, )
self.mov_prod(window=None, )
```

Note that `mov_avg` and `mov_mean` are identical functions and the names
can be used interchangeably.


### Input arguments ###


???+ input "self"
    Time series on whose data a moving function is applied:

    - `mov_sum`: moving sum
    - `mov_avg` or `mov_mean`: moving average
    - `mov_prod`: moving product

???+ input "window"
    Number of observations (a negative integer) to include in the moving
    function (counting from the current time period backwards). If
    `window=None`, the default window is derived from the time series
    frequency:

    | Date frequency | Default window |
    |----------------|---------------:|
    | Yearly         |             –1 |
    | Quarterly      |             –4 |
    | Monthly        |            –12 |
    | Weekly         |            –52 |
    | Daily          |           –365 |
    | Otherwise      |             –4 |

    Note that the default windows for daily and weekly frequencies are
    fixed number and do not depend on the actual number of days in a
    calendar year or the actual number of weeks in a calendar year.


### Returns ###


???+ returns "output"
    New time series with data calculated as the moving function of the
    original data.

???+ returns "self"
    Time series with data replaced by the moving function of the original
    data.

················································································
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


Inlay.mov_sum.__doc__ = Inlay._moving.__doc__
Inlay.mov_avg.__doc__ = Inlay._moving.__doc__
Inlay.mov_mean.__doc__ = Inlay._moving.__doc__
Inlay.mov_prod.__doc__ = Inlay._moving.__doc__


for n in ("mov_sum", "mov_avg", "mov_mean", "mov_prod", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )

