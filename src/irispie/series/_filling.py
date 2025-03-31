"""
"""


#[

from __future__ import annotations

import functools as _ft
import numpy as _np
import documark as _dm
from typing import Literal

from ._functionalize import FUNC_STRING

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import Any, Callable, EllipsisType
    from collections.abc import Iterable
    from numbers import Real
    from ..dates import Period
    from ..series import Series

#]


__all__ = []


MethodType = Literal["next", "previous", "nearest", "linear", "log_linear", "constant", "from_series", ]


class Inlay:
    """
    """
    #[

    @_dm.reference(category="homogenizing", )
    def fill_missing(
        self,
        method: Literal["next", "previous", "nearest", "linear", "log_linear", "constant", "from_series", ],
        method_args: Any | None = None,
        span: Iterable[Period] | EllipsisType | None = None,
    ) -> None:
        r"""
················································································

==Fill missing observations==


### Function form for creating new time `Series` objects ###

    new = irispie.fill_missing(
        self,
        method,
        *args,
        span=None,
    )


### Class method form changing existing `Series` objects in-place ###

    self.fill_missing(
        method,
        *args,
        span=None,
    )


### Input arguments ###


???+ input "self"
    The time `Series` object to be filled.

???+ input "method"
    The method to be used for filling missing observations. The following methods are available:

    | Method         | Description
    |----------------|-------------
    | "constant"     | Fill with a constant value
    | "next"         | Next available observation
    | "previous"     | Previous available observation
    | "nearest"      | Nearest available observation
    | "linear"       | Linear interpolation or extrapolation
    | "log_linear"   | Log-linear interpolation or extrapolation
    | "from_series"  | Fill with values from another time series object

???+ input "*args"
    Additional arguments to be passed to the filling method. The following methods require additional arguments:

    | Method         | Additional argument(s)
    |----------------|-----------------------
    | "constant"     | A single constant value
    | "from_series"  | A time `Series` object


???+ input "span"
    The time span to be filled. If `None`, the time span of the input time `Series` is filled.


### Returns ###


???+ returns "self"
    The time `Series` object with missing observations filled.

???+ returns "new"
    A new time `Series` object with missing observations filled.

················································································
        """
        fill_func = _FILL_METHOD_DISPATCH[method]
        data, span = self.get_data_and_periods(span, )
        new_data = [
            fill_func(variant, method_args, span=span, ).T
            for variant in data.T
        ]
        self.set_data(span, new_data, )

    #]


attributes = (n for n in dir(Inlay) if not n.startswith("_"))
for n in attributes:
    code = FUNC_STRING.format(n=n, )
    exec(code, globals(), locals(), )
    __all__.append(n)


def _fill_neighbor(
    data,
    method_args: Any | None,
    func: Callable,
    **kwargs,
) -> _np.ndarray:
    """
    """
    index_nan = _np.isnan(data)
    if _np.all(index_nan):
        return data
    where_obs = _np.where(~index_nan)[0].astype(_np.float64, )
    if where_obs.size == 0:
        return data
    where_nan = _np.where(index_nan)[0]
    for i in where_nan:
        j = func(i, where_obs, )
        data[i] = data[j] if j is not None else _np.nan
    return data


def _fill_interp(
    data: _np.ndarray,
    method_args: Any | None,
    func: Callable,
    **kwargs,
) -> _np.ndarray:
    """
    """
    index_nan = _np.isnan(data)
    if _np.all(index_nan):
        return data
    where_obs = _np.where(~index_nan)[0].astype(_np.float64, )
    if where_obs.size == 0:
        return data
    where_nan = _np.where(index_nan)[0]
    for i in where_nan:
        prev = _previous_index(i, where_obs, )
        next_ = _next_index(i, where_obs, )
        if prev is not None and next_ is not None:
            data[i] = func(data[prev], data[next_], prev, next_, i, )
            continue
        if prev is not None:
            data[i] = data[prev]
            continue
        if next_ is not None:
            data[i] = data[next_]
            continue
    return data


def _interpolation_linear(
    previous_value: Real,
    next_value: Real,
    previous_index: int,
    next_index: int,
    curr_index: int,
    /,
) -> Real:
    """
    """
    diff = next_value - previous_value
    scale_diff = (curr_index - previous_index) / (next_index - previous_index)
    return previous_value + diff * scale_diff


def _interpolation_log_linear(
    previous_value: Real,
    next_value: Real,
    previous_index: int,
    next_index: int,
    curr_index: int,
    /,
) -> Real:
    """
    """
    return _np.exp(_interpolation_linear(
        _np.log(previous_value),
        _np.log(next_value),
        previous_index,
        next_index,
        curr_index,
    ))


def _fill_constant(
    data: _np.ndarray,
    method_args: Real,
    **kwargs,
) -> _np.ndarray:
    """
    """
    constant = method_args
    index_nan = _np.isnan(data)
    data[index_nan] = constant
    return data


def fill_from_series(
    values: _np.ndarray,
    method_args: Series,
    span: Iterable[Period],
) -> _np.ndarray:
    """
    """
    series = method_args
    fill_values = series.get_data(span, )
    index_nan = _np.isnan(values)
    values[index_nan] = fill_values.flatten()[index_nan]
    return values


def _next_index(i: int, where_obs: _np.ndarray, /, ) -> int:
    """
    """
    #[
    where_obs_minus_i = where_obs - i
    index_negative = where_obs_minus_i < 0
    if _np.all(index_negative):
        return None
    where_obs_minus_i[index_negative] = _np.inf
    index = _np.argmin(where_obs_minus_i)
    return int(where_obs[index])
    #]


def _previous_index(i: int, where_obs: _np.ndarray, /, ) -> int:
    """
    """
    #[
    i_minus_where_obs = i - where_obs
    index_negative = i_minus_where_obs < 0
    if _np.all(index_negative):
        return None
    i_minus_where_obs[index_negative] = _np.inf
    index = _np.argmin(i_minus_where_obs)
    return int(where_obs[index])
    #]


def _nearest_index(i: int, where_obs: _np.ndarray, /, ) -> int:
    """
    """
    #[
    i_minus_where_obs = _np.abs(i - where_obs)
    index = _np.argmin(i_minus_where_obs)
    return int(where_obs[index])
    #]


_FILL_METHOD_DISPATCH = {
    "next": _ft.partial(_fill_neighbor, func=_next_index, ),
    "previous": _ft.partial(_fill_neighbor, func=_previous_index, ),
    "nearest": _ft.partial(_fill_neighbor, func=_nearest_index, ),
    "linear": _ft.partial(_fill_interp, func=_interpolation_linear, ),
    "log_linear": _ft.partial(_fill_interp, func=_interpolation_log_linear, ),
    "constant": _fill_constant,
    "from_series": fill_from_series,
    #
    # Aliases for backward compatibility
    "series": fill_from_series,
}


