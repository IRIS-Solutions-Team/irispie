"""
Time series mixin for moving sum, average, product
"""


#[

from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Callable, )
import numpy as _np
import functools as _ft
import documark as _dm

from .. import dates as _dates
from ._functionalize import FUNC_STRING

#]


__all__ = []


class Inlay:
    """
    Time series mixin for moving sum, average, product
    """
    #[

    @_dm.reference(
        category=None,
        call_name="Moving window calculations",
        call_name_is_code=False,
        priority=20,
    )
    def moving_window(
        self,
        func: Callable,
        /,
        window: int | None = None,
    ) -> Self:
        r"""
················································································


Overview of moving window functions:

| Function | Description
|----------|-------------
| `mov_sum` | Moving sum, $y_t = \sum_{i=0}^{k-1} x_{t-i}$
| `mov_avg` | Moving average, $y_t = \frac{1}{k} \sum_{i=0}^{k-1} x_{t-i}$
| `mov_mean` | Same as `mov_avg`
| `mov_prod` | Moving product, $y_t = \prod_{i=0}^{k-1} x_{t-i}$

where

* $k$ is the window length determined by the `window` input argument
(note that $k$ above is a positive integer while `window` needs to be
entered as a negative integer).



### Function for creating new Series objects ###

```
new = irispie.mov_sum(self, window=None, )
new = irispie.mov_avg(self, window=None, )
new = irispie.mov_mean(self, window=None, )
new = irispie.mov_prod(self, window=None, )
```


### Methods for changing existing Series objects in-place ###


```
self.mov_sum(window=None, )
self.mov_avg(window=None, )
self.mov_mean(window=None, )
self.mov_prod(window=None, )
```


### Input arguments ###


???+ input "self"
    Time series on whose data a moving function is calculated (see the
    overview table above).

???+ input "window"
    A negative interger determining the number of observations to include
    in the moving window, counting from the current time period backwards
    (the minus sign is a convention to indicate that the window goes
    backwards in time). If `window=None` (or not specified), the default
    window is derived from the time series frequency:

    | Date frequency | Default window |
    |----------------|---------------:|
    | `YEARLY`       |             –1 |
    | `QUARTERLY`    |             –4 |
    | `MONTHLY`      |            –12 |
    | `WEEKLY`       |            –52 |
    | `DAILY`        |           –365 |
    | Otherwise      |             –4 |

    Note that the default windows for `DAILY` and `WEEKLY` frequencies are
    fixed numbers and do not depend on the actual number of days in a
    calendar year or the actual number of weeks in a calendar year.


### Returns ###


???+ returns "new"
    New time series with data calculated as a moving window function of the
    original data.

???+ returns "self"
    Time series with data replaced by the moving window function of the
    original data.

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

    @_dm.reference(category="moving", )
    def mov_sum(self, window: int | None = None, ) -> Self:
        r"""
................................................................................

==Moving sum==


See documentation of [moving window calculations](#moving-window-calculations) or
`help(irispie.Series.moving_window)`.

................................................................................
        """
        return self.moving_window(_np.sum, window=window, )

    @_dm.reference(category="moving", )
    def mov_avg(self, window: int | None = None, ) -> Self:
        r"""
................................................................................

==Moving average==


See documentation of [moving window calculations](#moving-window-calculations) or
`help(irispie.Series.moving_window)`.

................................................................................
        """
        return self.moving_window(_np.mean, window=window, )

    @_dm.reference(category="moving", )
    def mov_mean(self, window: int | None = None, ) -> Self:
        r"""
................................................................................

==Moving average==


See documentation of [moving window calculations](#moving-window-calculations) or
`help(irispie.Series.moving_window)`.

................................................................................
        """
        return self.moving_window(_np.mean, window=window, )

    @_dm.reference(category="moving", )
    def mov_prod(self, window: int | None = None, ) -> Self:
        r"""
................................................................................

==Moving product==


See documentation of [moving window calculations](#moving-window-calculations) or
`help(irispie.Series.moving_window)`.

................................................................................
        """
        return self.moving_window(_np.prod, window=window, )

    def _get_default_moving_window(self, /, ) -> int:
        r"""
        Derive default moving window from time series frequency
        """
        return (
            -self.frequency.value
            if self.frequency is not None and self.frequency.value > 0
            else -4
        )

    #]


attributes = (n for n in dir(Inlay) if not n.startswith("_"))
for n in attributes:
    code = FUNC_STRING.format(n=n, )
    exec(code, globals(), locals(), )
    __all__.append(n)

