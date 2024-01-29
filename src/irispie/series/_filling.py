"""
"""


#[
from __future__ import annotations

import functools as _ft
import numpy as _np

from .. import dates as _dates
from . import _functionalize
#]


__all__ = ()


class Inlay:
    """
    """
    #[

    def fill_missing(
        self,
        fill_range: Iterable[_dates.Dater],
        method: Literal["next", "previous", "nearest", "constant"],
        *args,
        **kwargs,
    ) -> None:
        """
        """
        fill_func = _METHOD_FACTORY[method]
        data = self.get_data(fill_range, )
        new_data = [
            fill_func(variant, *args, **kwargs, ).T
            for variant in data.T
        ]
        self.set_data(fill_range, new_data, )

    #]


for n in ("fill_missing", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _fill_neighbor(
    data,
    func,
    *args,
    **kwargs,
) -> _np.ndarray:
    """
    """
    index_nan = _np.isnan(data)
    if _np.all(index_nan):
        return data
    where_nan = _np.where(index_nan)[0]
    where_obs = _np.where(~index_nan)[0].astype(_np.float64, )
    if where_obs.size == 0:
        return data
    for i in where_nan:
        j = func(i, where_obs, )
        data[i] = data[int(j)] if j is not None else _np.nan
    return data


def _fill_constant(
    data,
    constant,
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
    return where_obs[index]
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
    return where_obs[index]
    #]


def _nearest_index(i: int, where_obs: _np.ndarray, /, ) -> int:
    """
    """
    #[
    i_minus_where_obs = _np.abs(i - where_obs)
    index = _np.argmin(i_minus_where_obs)
    return where_obs[index]
    #]


_METHOD_FACTORY = {
    "next": _ft.partial(_fill_neighbor, func=_next_index, ),
    "previous": _ft.partial(_fill_neighbor, func=_previous_index, ),
    "nearest": _ft.partial(_fill_neighbor, func=_nearest_index, ),
    "constant": _fill_constant,
}


