"""
Time series
"""

#[
from __future__ import annotations
from IPython import embed

from numbers import Number
from collections.abc import Iterable, Callable
from typing import Self, TypeAlias, NoReturn
from types import EllipsisType
import numpy as np_
import itertools as it_
import copy as co_

from .dates import (Dater, Ranger, date_index, ResolvableProtocol, )
#]


__all__ = [
    "Series",
]


underscore_functions = ["log", "exp", "sqrt", "max", "min", "mean", "median"]


Dates: TypeAlias = Dater | Iterable[Dater] | Ranger | EllipsisType | None
Columns: TypeAlias = int | Iterable[int] | slice
Data: TypeAlias = Number | Iterable[Number] | tuple | np_.ndarray


_VERTICAL_ELLIPSIS = "⋮"


def _str_row(date, data, date_str_format, numeric_format, nan_str: str):
    date_str = ("{:"+date_str_format+"}").format(date)
    value_format = "{:"+numeric_format+"}"
    data_str = "".join([value_format.format(v) for v in data])
    data_str = data_str.replace("nan", "{:>3}".format(nan_str))
    return date_str + data_str


def _short_str_row(shape, date_str_format, short_str_format):
    date_str = ("{:"+date_str_format+"}").format(_VERTICAL_ELLIPSIS)
    value_format = "{:"+short_str_format+"}"
    data_str = "".join([value_format.format(_VERTICAL_ELLIPSIS)] * shape[1])
    return date_str + data_str


def _get_date_positions(dates, base, num_periods):
    pos = list(date_index(dates, base))
    min_pos = min(pos)
    max_pos = max(pos)
    add_before = max(-min_pos, 0)
    add_after = max(max_pos - num_periods, 0)
    pos_adjusted = [p + add_before for p in pos]
    return pos_adjusted, add_before, add_after


def _trim_decorate(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self._trim()
    return wrapper


class Series:
    """
    """
    __slots__ = [
        "start_date", "data", "data_type",
        "_comment", "_column_names", "_user_data", "_short_rows"
    ]
    numeric_format: str = "15g"
    short_str_format: str = ">15"
    date_str_format: str = ">12"
    nan_str: str = "·"

    def __init__(self, /, num_columns=1, data_type=float, ):
        self.data_type = data_type
        self.reset(num_columns=num_columns, data_type=data_type)

    def reset(self, num_columns=None, data_type=None):
        num_columns = num_columns if num_columns else self.num_columns
        data_type = data_type if data_type else self.data_type
        self.start_date = None
        self.data = np_.full((0, num_columns), np_.nan, dtype=data_type)
        self._comment = ""
        self._column_names = [""] * num_columns
        self._user_data = {}
        self._short_rows = 5
        return self

    def _create_rows_of_nans(self, num_rows=0):
        return np_.full((num_rows, self.shape[1]), np_.nan, dtype=self.data_type)

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_columns(self):
        return self.data.shape[1]

    @property
    def range(self):
        return Ranger(self.start_date, self.end_date) if self.start_date else None

    @property
    def end_date(self):
        return self.start_date + self.data.shape[0] - 1 if self.start_date else None

    @classmethod
    def from_data(cls, dates, data):
        self = cls()
        self.set_data(dates, data)

    @_trim_decorate
    def set_data(
        self,
        dates: Dates,
        data: Data,
        columns: Columns = None,
        /,
    ) -> Self:
        dates = self._resolve_dates(dates)
        columns = self._resolve_columns(columns)
        if not self.start_date:
            self.start_date = next(iter(dates))
            self.data = self._create_rows_of_nans(1)
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before
        if isinstance(data, np_.ndarray):
            self.data[pos, :] = data
            return
        if not isinstance(data, tuple):
            data = it_.repeat(data)
        for c, d in zip(columns, data):
            self.data[pos, c] = d
        return self

    def get_data(
        self,
        dates: Dates,
        columns: Columns = None,
        /,
    ) -> np_.ndarray:
        """
        """
        dates = self._resolve_dates(dates)
        columns = self._resolve_columns(columns)
        base_date = self.start_date
        if not dates or not base_date:
            num_dates = len(set(dates))
            return self._create_rows_of_nans(num_dates)[:,columns]
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after)
        if not isinstance(pos, Iterable):
            pos = (pos, )
        return data[np_.ix_(pos, columns)]

    def get_data_column(
        self,
        dates: Dates,
        column: Number | None = None,
        /,
    ) -> np_.ndarray:
        """
        """
        column = column if column and column<self.data.shape[1] else 0
        return self.get_data(dates, column)

    def set_start_date(
        self,
        new_start_date: Dater,
        /,
    ) -> Self:
        self.start_date = new_start_date
        return self

    def _resolve_dates(self, dates):
        if dates is None:
            return []
        if dates is Ellipsis:
            dates = Ranger(None, None)
        if isinstance(dates, ResolvableProtocol) and dates.needs_resolve:
            dates = dates.resolve(self)
        return dates

    def _resolve_columns(self, columns):
        if columns is None or columns is Ellipsis:
            columns = slice(None)
        if isinstance(columns, slice):
            columns = range(*columns.indices(self.num_columns))
        if not isinstance(columns, Iterable):
            columns = (columns, )
        return columns

    def __call__(self, shift):
        return co_.deepcopy(self).set_start_date(
            self.start_date - shift if self.start_date is not None else None
        )

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, None, )
        return self.get_data(index[0], index[1])

    def __setitem__(self, index, data):
        if not isinstance(index, tuple):
            index = (index, None, )
        return self.set_data(index[0], data, index[1])

    def hstack(self, *args):
        if not args:
            return co_.deepcopy(self)
        encompassing_range = self._get_encompassing_range(*args)
        new_data = self.get_data(encompassing_range)
        add_data = (
            x.get_data(encompassing_range) if hasattr(x, "get_data") else _create_data_from_number(x, encompassing_range, self.data_type)
            for x in args
        )
        new_data = np_.hstack((new_data, *add_data))
        new = Series(num_columns=new_data.shape[1]);
        new.set_data(encompassing_range, new_data)
        return new

    def overlay_by_range(
        self,
        other: Self,
        /,
    ) -> Self:
        self._trim()
        other._trim()
        self.set_data(other.range, other.data, )
        return self

    def overlay(
        self,
        other: Self,
        /,
        method = "by_range",
    ) -> Self:
        return self._LAY_METHOD_RESOLUTION[method]["overlay"](self, other, )

    def underlay_by_range(
        self,
        other: Self,
        /,
    ) -> Self:
        """
        """
        self._trim()
        other._trim()
        self_range = self.range
        self_data = self.data
        self.start_date = other.start_date
        self.data = other.data
        self.set_data(self_range, self_data, )
        return self

    def underlay(
        self,
        other: Self,
        /,
        method = "by_range",
    ) -> Self:
        return self._LAY_METHOD_RESOLUTION[method]["underlay"](self, other, )

    _LAY_METHOD_RESOLUTION = {
        "by_range": {"overlay": overlay_by_range, "underlay": underlay_by_range},
    }

    def __or__(self, other):
        return self.hstack(other)

    def __lshift__(self, other):
        return co_.deepcopy(self).overlay_by_range(other, )

    def __rshift__(self, other):
        return co_.deepcopy(other).overlay_by_range(self, )

    def _get_encompassing_range(*args) -> Ranger:
        start_dates = [x.start_date for x in args if hasattr(x, "start_date") and x.start_date]
        end_dates = [x.end_date for x in args if hasattr(x, "end_date") and x.end_date]
        start_date = min(start_dates) if start_dates else None
        end_date = max(end_dates) if end_dates else None
        return Ranger(start_date, end_date)

    def _trim(self):
        if self.data.size==0:
            return self.reset()
        num_leading = _get_num_leading_nan_rows(self.data)
        if num_leading == self.data.shape[0]:
            return self.reset()
        num_trailing = _get_num_leading_nan_rows(self.data[::-1])
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading if num_leading else None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to), ...]
        if slice_from:
            self.start_date += int(slice_from)
        return self

    def _create_expanded_data(self, add_before, add_after):
        data = np_.copy(self.data)
        if add_before:
            data = np_.vstack((self._create_rows_of_nans(add_before), data))
        if add_after:
            data = np_.vstack((data, self._create_rows_of_nans(add_after)))
        return data

    def _get_header_view(self, /, ):
        shape = self.data.shape
        return [ 
            "", 
            f"Series {shape[0]}-by-{shape[1]} {self.start_date}:{self.end_date}",
            f"Comment: \"{self._comment}\"",
            "", 
        ]

    def _get_data_view(self, /, ):
        """
        """
        return [
            _str_row(date, data_row, self.date_str_format, self.numeric_format, self.nan_str) 
            for row, (date, data_row) in enumerate(zip(self.range, self.data))
        ]

    def __str__(self, /, ):
        return "\n".join(self._get_header_view() + self._get_data_view())

    def __repr__(self, /, ):
        return self.__str__()

    def _check_data_shape(self, data, /, ):
        if data.shape[1] != self.data.shape[1]:
            raise Exception("Time series data being assigned must preserve the number of columns")

    def __neg__(self):
        """
        -self
        """
        output = co_.deepcopy(self)
        output.data = -output.data
        return output

    def __pos__(self):
        """
        +self
        """
        return co_.deepcopy(self.copy)

    def __invert__(self):
        """
        ~self for short display
        """
        header_view = self._get_header_view()
        data_view = self._get_data_view()
        short = _short_str_row(self.data.shape, self.date_str_format, self.short_str_format, )
        data_view = data_view[:self._short_rows] + [short] + data_view[-self._short_rows:]
        print("\n".join(header_view + data_view))

    def __add__(self, other) -> Self|np_.ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__add__(arg))

    def __radd__(self, other):
        """
        other + self
        """
        return self._unop(lambda data: data.__radd__(other))

    def __mul__(self, other) -> Self|np_.ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__mul__(arg))

    def __rmul__(self, other):
        """
        other + self
        """
        return self._unop(lambda data: data.__rmul__(other))

    def __sub__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__sub__(arg))

    def __rsub__(self, other):
        """
        other - self
        """
        return self._unop(lambda data: data.__rsub__(other))

    def __pow__(self, other):
        """
        self ** other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__pow__(arg))

    def __rpow__(self, other):
        """
        other ** self
        """
        return self._unop(lambda data: data.__rpow__(other))

    def __truediv__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__truediv__(arg))

    def __rtruediv__(self, other):
        """
        other - self
        """
        return self._unop(lambda data: data.__rtruediv__(other))

    def __floordiv__(self, other):
        """
        self // other
        """
        return self._unop(lambda data: data.__floordiv__(other))

    def __rfloordiv__(self, other):
        """
        other // self
        """
        return self._unop(lambda data: data.__rfloordiv__(other))

    def __mod__(self, other):
        """
        self % other
        """
        return self._unop(lambda data: data.__mod__(other))

    def __rmod__(self, other, /, ):
        """
        other % self
        """
        return self._unop(lambda data: data.__rmod__(other))

    def _unop(self, func, /, *args, **kwargs):
        new_data = func(self.data, *args, **kwargs)
        axis = kwargs.get("axis")
        if (axis is None or axis==1) and new_data.shape==self.data.shape:
            new = co_.deepcopy(self)
            new.data = new_data
            return new._trim()
        elif (axis is None or axis==1) and new_data.shape==(self.data.shape[0],):
            new = co_.deepcopy(self)
            new.data = new_data.reshape(self.data.shape[0], 1)
            return new._trim()
        elif axis is None and new_data.shape==():
            return new_data
        elif axis==0 and new_data.shape==(self.data.shape[1],):
            return new_data
        else:
            raise Exception("Unary operation on a time series resulted in a data array with an unexpected shape")


    def _binop(self, other, func, /, ):
        if not isinstance(other, type(self)):
            unop_func = lambda data: func(data, other)
            return _unop(self, unop_func)


    def _replace_data(self, new_data, /, ):
        self.data = new_data
        return self._trim()

    def copy(self, /, ) -> Self:
        return co_.deepcopy(self)

    for n in ["log", "exp", "sqrt"]:
        exec(f"def {n}(self): return self._replace_data(np_.{n}(self.data))")

    for n in underscore_functions:
        exec(f"def _{n}_(self, *args, **kwargs): return self._unop(func=np_.{n}, *args, **kwargs)")

def _get_num_leading_nan_rows(data):
    try:
        num = next(x[0] for x in enumerate(data) if not np_.all(np_.isnan(x[1])))
    except StopIteration:
        num = data.shape[0]
    return num


def _unop(x, func: Callable, *args, **kwargs) -> object:
    if isinstance(x, Series):
        return x._unop(func, *args, **kwargs)
    else:
        return func(x, *args, **kwargs)


def hstack(first, *args) -> Self:
    return first.hstack(*args)


def _create_data_from_number(
    number: Number,
    range: Ranger,
    data_type: type,
    /,
) -> np_.ndarray:
    return np_.full((len(range), 1), number, dtype=data_type)

