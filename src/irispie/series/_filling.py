"""
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, Literal, )
import functools as _ft
import numpy as _np

from .. import pages as _pages
from . import _functionalize

if TYPE_CHECKING:
    from typing import (Callable, EllipsisType, )
    from collections.abc import (Iterable, )
    from numbers import (Real, )
    from ..data import (Period, )
#]


__all__ = ()


class Inlay:
    """
    """
    #[

    @_pages.reference(category="homogenizing", )
    def fill_missing(
        self,
        method: Literal["next", "previous", "nearest", "linear", "log_linear", "constant"],
        span: Iterable[Period] | EllipsisType | None = None,
        *args,
        **kwargs,
    ) -> None:
        r"""
················································································

==Fill missing observations==


### Function form for creating new time `Series` objects ###

    new = irispie.fill_missing(
        self,
        method,
        span=None,
        *args,
    )


### Class method form for changing existing time `Series` objects in-place ###

    self.fill_missing(
        method,
        span=None,
        *args,
    )


### Input arguments ###


???+ input "self"
    The time `Series` object to be filled.

???+ input "method"
    The method to be used for filling missing observations. The following methods are available:

    | Method        | Description
    |---------------|-------------
    | "next"        | Next available observation
    | "previous"    | Previous available observation
    | "nearest"     | Nearest available observation
    | "linear"      | Linear interpolation or extrapolation
    | "log_linear"  | Log-linear interpolation or extrapolation

???+ input "span"
    The time span to be filled. If `None`, the time span of the input time `Series` is filled.


### Returns ###


???+ returns "self"
    The time `Series` object with missing observations filled.

???+ returns "new"
    A new time `Series` object with missing observations filled.

················································································
        """
        fill_func = _METHOD_FACTORY[method]
        data = self.get_data(span, )
        new_data = [
            fill_func(variant, *args, **kwargs, ).T
            for variant in data.T
        ]
        self.set_data(span, new_data, )

    #]


for n in ("fill_missing", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _fill_neighbor(
    data,
    /,
    func: Callable,
    *args,
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
    /,
    func: Callable,
    *args,
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
    constant: Real,
    /,
    *args,
    **kwargs,
) -> _np.ndarray:
    """
    """
    index_nan = _np.isnan(data)
    data[index_nan] = constant
    return data


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


_METHOD_FACTORY = {
    "next": _ft.partial(_fill_neighbor, func=_next_index, ),
    "previous": _ft.partial(_fill_neighbor, func=_previous_index, ),
    "nearest": _ft.partial(_fill_neighbor, func=_nearest_index, ),
    "linear": _ft.partial(_fill_interp, func=_interpolation_linear, ),
    "log_linear": _ft.partial(_fill_interp, func=_interpolation_log_linear, ),
    "constant": _fill_constant,
}


