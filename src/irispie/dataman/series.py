"""
Time series
"""

#[
from __future__ import annotations
# from IPython import embed

from numbers import Number
from collections.abc import Iterable, Callable
from typing import Self, TypeAlias, NoReturn
from types import EllipsisType
import numpy as np_
import itertools as it_
import copy as co_

from ..dataman.dates import (Dater, Ranger, date_index, ResolvableProtocol, )
from ..dataman import (views as vi_, )
from ..dataman import (dates as da_, )
from ..dataman import (filters as fi_, )
from ..dataman import (plotly as pl_, )
from ..mixins import (userdata as ud_, )
#]


FUNCTION_ADAPTATIONS = ["log", "exp", "sqrt", "maximum", "minimum"] + ["cumsum"] + ["maxi", "mini", "mean", "median"]
np_.maxi = np_.max
np_.mini = np_.min


__all__ = [
    "Series", "shift", "diff", "difflog", "pct", "roc",
] + FUNCTION_ADAPTATIONS


for n in FUNCTION_ADAPTATIONS:
    exec(
        f"def {n}(x, *args, **kwargs): "
        f"return x._{n}_(*args, **kwargs) if hasattr(x, '_{n}_') else np_.{n}(x, *args, **kwargs)"
    )


Dates: TypeAlias = Dater | Iterable[Dater] | Ranger | EllipsisType | None
Columns: TypeAlias = int | Iterable[int] | slice
Data: TypeAlias = Number | Iterable[Number] | tuple | np_.ndarray


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


class Series(fi_.HodrickPrescottMixin, pl_.PlotlyMixin, ud_.DescriptionMixin, vi_.SeriesViewMixin):
    """
    """
    __slots__ = (
        "start_date", "data", "data_type",
        "_description", "_column_titles", "_user_data",
    )
    _numeric_format: str = "15g"
    _short_str_format: str = ">15"
    _date_str_format: str = ">12"
    _missing = np_.nan
    _missing_str: str = "·"
    _test_missing_period = staticmethod(lambda x: np_.all(np_.isnan(x)))

    def __init__(self, /, *args, **kwargs):
        num_columns = kwargs.get("num_columns", 1)
        self.data_type = kwargs.get("data_type", np_.float64)
        self._description = kwargs.get("description", "")
        self.reset(num_columns)

    def reset(self, /, num_columns=None, data_type=None, description=None, ):
        num_columns = num_columns if num_columns else self.num_columns
        data_type = data_type if data_type else self.data_type
        description = description if description else self._description
        self.start_date = None
        self.data = np_.full((0, num_columns), self._missing, dtype=data_type)
        self._description = str(description)
        self._column_titles = [""] * num_columns
        self._user_data = {}
        return self

    def _create_periods_of_missing_values(self, num_rows=0):
        return np_.full((num_rows, self.shape[1]), self._missing, dtype=self.data_type)

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_columns(self):
        return self.data.shape[1]

    @property
    def range(self):
        return Ranger(self.start_date, self.end_date) if self.start_date else []

    @property
    def end_date(self):
        return self.start_date + self.data.shape[0] - 1 if self.start_date else None

    @property
    def frequency(self):
        return self.start_date.frequency if self.start_date is not None else da_.Frequency.UNKNOWN

    @classmethod
    def from_dates_and_data(
        cls,
        dates: Iterable[Dater],
        data: np_.ndarray,
        /,
    ) -> Self:
        num_columns = data.shape[1] if hasattr(data, "shape") else 1
        self = cls(num_columns=num_columns)
        self.set_data(dates, data)
        return self

    @classmethod
    def from_start_date_and_data(
        cls,
        start_date: Dater,
        data: np_.ndarray,
        /,
        **kwargs,
    ) -> Self:
        num_columns = data.shape[1]
        self = cls(num_columns=num_columns, **kwargs)
        self.start_date = start_date
        self.data = data
        return self

    @classmethod
    def from_func(
        cls,
        dates: Iterable[Dater],
        func: Callable,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        self = cls(**kwargs)
        dates = [ t for t in dates ]
        data = [ [ func() for j in range(self.num_columns) ] for i in range(len(dates)) ]
        data = np_.array(data, dtype=self.data_type)
        self.set_data(dates, data)
        return self

    @_trim_decorate
    def set_data(
        self,
        dates: Dates,
        data: Data | Series,
        columns: Columns = None,
        /,
    ) -> Self:
        dates = self._resolve_dates(dates)
        columns = self._resolve_columns(columns)
        if not self.start_date:
            self.start_date = next(iter(dates))
            self.data = self._create_periods_of_missing_values(1)
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before
        if isinstance(data, Series):
            data = data.get_data(dates)
        if isinstance(data, np_.ndarray):
            self.data[pos, :] = data
            return
        if not isinstance(data, tuple):
            data = it_.repeat(data)
        for c, d in zip(columns, data):
            self.data[pos, c] = d
        return self

    def _get_data_and_recreate(
        self,
        *args,
    ) -> np_.ndarray:
        """
        """
        data, dates = self.get_data_and_resolved_dates(*args)
        num_columns = data.shape[1]
        new = Series(num_columns=num_columns, data_type=self.data_type)
        new.set_data(dates, data)
        return new

    def get_data_and_resolved_dates(
        self,
        dates: Dates,
        columns: Columns = None,
        /,
    ) -> np_.ndarray:
        """
        """
        dates = [ t for t in self._resolve_dates(dates) ]
        columns = self._resolve_columns(columns)
        base_date = self.start_date
        if not dates or not base_date:
            num_dates = len(set(dates))
            return self._create_periods_of_missing_values(num_dates)[:, columns], dates
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after)
        if not isinstance(pos, Iterable):
            pos = (pos, )
        return data[np_.ix_(pos, columns)], dates

    def get_data(
        self,
        *args,
    ) -> np_.ndarray:
        return self.get_data_and_resolved_dates(*args)[0]

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

    def extract_columns(
        self,
        columns,
        /,
    ) -> NoReturn:
        if not isinstance(columns, Iterable):
            columns = (columns, )
        columns = [ c for c in columns ]
        self.data = self.data[:, columns]
        self.column_titles = [ self.column_titles[c] for c in columns ]

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

    def __call__(self, *args):
        """
        Get data self[dates] or self[dates, columns]
        """
        return self.get_data(*args)

    def _shift(self, shift_by) -> Self:
        """
        Shift (lag, lead) start date
        """
        self.start_date = (
            self.start_date - shift_by 
            if self.start_date else self.start_date
        )

    def __getitem__(self, index):
        """
        Create a new time series based on date retrieved by self[dates] or self[dates, columns]
        """
        if isinstance(index, int):
            new = co_.deepcopy(self)
            new._shift(index)
            return new
        if not isinstance(index, tuple):
            index = (index, None, )
        return self._get_data_and_recreate(*index)

    def __setitem__(self, index, data):
        """
        Set data self[dates] = ... or self[dates, columns] = ...
        """
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

    def clip(
        self,
        new_start_date: Dater | None,
        new_end_date: Dater | None,
    ) -> NoReturn:
        if new_start_date is None or new_start_date < self.start_date:
            new_start_date = self.start_date
        if new_end_date is None or new_end_date > self.end_date:
            new_end_date = self.end_date
        if new_start_date == self.start_date and new_end_date == self.end_date:
            return
        self.start_date = new_start_date
        self.data = self.get_data(Ranger(new_start_date, new_end_date))

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
        #
        self.start_date = other.start_date
        enforce_columns = np_.zeros((1, self.data.shape[1]), dtype=self.data_type)
        self.data = enforce_columns + other.data
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
        num_leading = _get_num_leading_missing_rows(self.data, self._test_missing_period)
        if num_leading == self.data.shape[0]:
            return self.reset()
        num_trailing = _get_num_leading_missing_rows(self.data[::-1], self._test_missing_period)
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading if num_leading else None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to), ...]
        if slice_from:
            self.start_date += int(slice_from)
        return self

    def _create_expanded_data(self, add_before, add_after):
        return np_.pad(
            self.data, ((add_before, add_after), (0, 0)),
            mode="constant", constant_values=self._missing
        )

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

    def __add__(self, other) -> Self|np_.ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        """
        other + self
        """
        return self.apply(lambda data: data.__radd__(other))

    def __mul__(self, other) -> Self|np_.ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        """
        other + self
        """
        return self.apply(lambda data: data.__rmul__(other))

    def __sub__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        """
        other - self
        """
        return self.apply(lambda data: data.__rsub__(other))

    def __pow__(self, other):
        """
        self ** other
        """
        # FIXME: 1 ** numpy.nan -> 1 !!!!!
        arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, lambda x, y: x ** y)

    def __rpow__(self, other):
        """
        other ** self
        """
        return self.apply(lambda data: data.__rpow__(other))

    def __truediv__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        """
        other - self
        """
        return self.apply(lambda data: data.__rtruediv__(other))

    def __floordiv__(self, other):
        """
        self // other
        """
        return self._binop(other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        """
        other // self
        """
        return self.apply(lambda data: data.__rfloordiv__(other))

    def __mod__(self, other):
        """
        self % other
        """
        return self._binop(other, lambda x, y: x % y)

    def __rmod__(self, other, /, ):
        """
        other % self
        """
        return self.apply(lambda data: data.__rmod__(other))

    def apply(self, func, /, *args, **kwargs):
        new_data = func(self.data, *args, **kwargs)
        axis = kwargs.get("axis")
        if new_data.shape==self.data.shape:
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
            raise Exception("Function applied on a time series resulted in a data array with an unexpected shape")

    def _binop(self, other, func, /, ):
        if not isinstance(other, type(self)):
            unop_func = lambda data: func(data, other)
            return apply(self, unop_func)
        encompassing_range = self._get_encompassing_range(self, other)
        self_data = self.get_data(encompassing_range)
        other_data = other.get_data(encompassing_range)
        new = Series()
        new.start_date = encompassing_range[0]
        new.data = func(self_data, other_data)
        new._trim()
        return new

    def _replace_data(self, new_data, /, ):
        self.data = new_data
        return self._trim()

    def copy(self, /, ) -> Self:
        return co_.deepcopy(self, )

    for n in ["log", "exp", "sqrt", "maximum", "minimum"]:
        exec(f"def {n}(self, *args, **kwargs, ): return self._replace_data(np_.{n}(self.data, *args, **kwargs, ), )")
        exec(f"def _{n}_(self, *args, **kwargs, ): return self.apply(np_.{n}, *args, **kwargs, )")

    for n in ["cumsum"]:
        exec(f"def {n}(self): return self._replace_data(np_.{n}(self.data, axis=0, ), )")

    for n in ["maxi", "mini", "mean", "median"]:
        exec(f"def _{n}_(self, *args, **kwargs): return self.apply(np_.{n}, *args, **kwargs, )")


def _get_num_leading_missing_rows(data, test_missing_period, /, ):
    try:
        num = next(
            i for i, period_data in enumerate(data) 
            if not test_missing_period(period_data)
        )
    except StopIteration:
        num = data.shape[0]
    return num


# def apply(x, func: Callable, /, *args, **kwargs) -> object:
    # if isinstance(x, Series):
        # return x.apply(func, *args, **kwargs)
    # else:
        # return func(x, *args, **kwargs)


def hstack(first, *args) -> Self:
    return first.hstack(*args)


def _create_data_from_number(
    number: Number,
    range: Ranger,
    data_type: type,
    /,
) -> np_.ndarray:
    return np_.full((len(range), 1), number, dtype=data_type)


def shift(x, by):
    new = co_.deepcopy(x)
    new._shift(by)
    return new


def diff(x, by=-1):
    return x - shift(x, by)


def difflog(x, by=-1):
    return log(x) - log(shift(x, by))


def pct(x, shift=-1):
    return 100*(x/shift(x, by) - 1)


def roc(x, shift=-1):
    return x/shift(x, by)


