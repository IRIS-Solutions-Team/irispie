"""
Main time series class definition
"""


#[
from __future__ import annotations

from numbers import Number
from collections.abc import (Iterable, Callable, )
from typing import (Self, Any, TypeAlias, NoReturn, )
from types import (EllipsisType, )
import numpy as _np
import itertools as _it
import functools as _ft
import operator as _op
import copy as _cp

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from ..conveniences import iterators as _iterators
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings

from . import _diffcums
from . import _fillings
from . import _movings
from . import _conversions
from . import _hp
from . import _x13

from . import _plotly
from . import _views
from . import _functionalize

from ._diffcums import __all__ as _diffcums__all__
from ._diffcums import *

from ._fillings import __all__ as _fillings__all__
from ._fillings import *

from ._hp import __all__ as _hp__all__
from ._hp import *

from ._movings import __all__ as _movings__all__
from ._movings import *

from ._conversions import __all__ as _conversions__all__
from ._conversions import *
#]


FUNCTION_ADAPTATIONS_NUMPY = ("log", "exp", "sqrt", "maximum", "minimum", "mean", "median", )
FUNCTION_ADAPTATIONS_BUILTINS = ("max", "min", )
FUNCTION_ADAPTATIONS = tuple(set(FUNCTION_ADAPTATIONS_NUMPY + FUNCTION_ADAPTATIONS_BUILTINS))


__all__ = (
    ("Series", "shift", )
    + _conversions__all__
    + _diffcums__all__
    + _fillings__all__
    + _hp__all__
    + _movings__all__
    + FUNCTION_ADAPTATIONS
)

Dates: TypeAlias = _dates.Dater | Iterable[_dates.Dater] | _dates.Ranger | EllipsisType | None
ColumnsRequestType: TypeAlias = int | Iterable[int] | slice | None


def _get_date_positions(dates, base, num_periods):
    pos = tuple(_dates.date_index(dates, base))
    min_pos = _builtin_min((x for x in pos if x is not None), default=0)
    max_pos = _builtin_max((x for x in pos if x is not None), default=0)
    add_before = _builtin_max(-min_pos, 0)
    add_after = _builtin_max(max_pos - num_periods, 0)
    pos_adjusted = [
        p + add_before if p is not None else None
        for p in pos
    ]
    return pos_adjusted, add_before, add_after


def _trim_decorate(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.trim()
        return self
    return wrapper


class Series(
    _conversions.ConversionMixin,
    _diffcums.CumMixin,
    _diffcums.DiffMixin,
    _fillings.FillingMixin,
    _hp.Mixin,
    _movings.MovingMixin,
    _x13.Mixin,
    _plotly.Mixin,
    _descriptions.DescriptionMixin,
    _views.ViewMixin,
    _copies.CopyMixin,
):
    """
    Time series objects
    """
    #[

    __slots__ = (
        "start_date", "data", "data_type", "_column_titles",
        "metadata", "_description", 
    )

    _numeric_format: str = "15g"
    _short_str_format: str = ">15"
    _date_str_format: str = ">12"
    _missing = _np.nan
    _missing_str: str = "⋅"
    _test_missing_period = staticmethod(lambda x: _np.all(_np.isnan(x)))

    def __init__(
        self,
        /,
        *,
        num_columns: int = 1,
        data_type: type = _np.float64,
        description: str = "",
        column_titles: Iterable[str] | None = None,
        start_date: _dates.Dater | None = None,
        dates: Iterable[_dates.Dater] | None = None,
        values: Any | None = None,
        func: Callable | None = None,
    ) -> None:
        self.start_date = None
        self.data_type = data_type
        self.data = _np.full((0, num_columns), self._missing, dtype=self.data_type)
        self.metadata = {}
        self._description = description
        self.column_titles = column_titles
        self._reset_column_titles_if_needed()
        test = (x is not None for x in (start_date, dates, values, func))
        _SERIES_FACTORY.get(tuple(test), _invalid_constructor)(self, start_date=start_date, dates=dates, values=values, func=func, )

    def reset(self, /, ) -> None:
        self.__init__(
            num_columns=self.num_columns,
            data_type=self.data_type,
        )

    def _create_periods_of_missing_values(self, num_rows=0) -> _np.ndarray:
        """
        """
        return _np.full(
            (num_rows, self.shape[1], ),
            self._missing,
            dtype=self.data_type
        )

    @property
    def shape(self, /, ) -> tuple[int, int]:
        return self.data.shape

    @property
    def num_columns(self, /, ) -> int:
        return self.data.shape[1]

    @property
    def num_variants(self, /, ) -> int:
        return self.num_columns

    @property
    def range(self):
        return _dates.Ranger(self.start_date, self.end_date, ) if self.start_date else ()

    @property
    def dates(self, /, ) -> tuple[_dates.Dater, ...]:
        return tuple(self.range, )

    @property
    def end_date(self):
        return self.start_date + self.data.shape[0] - 1 if self.start_date else None

    @property
    def frequency(self):
        return (
            self.start_date.frequency
            if self.start_date is not None
            else _dates.Frequency.UNKNOWN
        )

    @classmethod
    def from_dates_and_values(
        klass,
        dates: Dates,
        values: Any,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        return klass(
            dates=dates,
            values=values,
            **kwargs,
        )

    @classmethod
    def from_start_date_and_values(
        klass,
        start_date: _dates.Dater,
        values: _np.ndarray | Iterable,
        /,
        **kwargs,
    ) -> Self:
        return klass(
            start_date=start_date,
            values=values,
            **kwargs,
        )

    @classmethod
    def from_dates_and_func(
        klass,
        dates: Iterable[_dates.Dater],
        func: Callable,
        /,
        **kwargs,
    ) -> Self:
        return klass(
            dates=dates,
            func=func,
            **kwargs,
        )

    @_trim_decorate
    def set_data(
        self,
        dates: Dates,
        data: Any | Series,
        columns: ColumnsRequestType = None,
        /,
    ) -> None:
        dates = self._resolve_dates(dates)
        columns = self._resolve_columns(columns)
        if not self.start_date:
            self.start_date = next(iter(dates), None, )
            self.data = self._create_periods_of_missing_values(num_rows=1, )
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before
        if hasattr(data, "get_data"):
            data = data.get_data(dates)
        if isinstance(data, _np.ndarray):
            self.data[pos, :] = data
            return
        if not isinstance(data, tuple):
            data = _it.repeat(data)
        for c, d in zip(columns, data):
            self.data[pos, c] = d

    def _get_data_and_recreate(
        self,
        *args,
    ) -> _np.ndarray:
        """
        """
        dates, pos, columns, expanded_data = self._resolve_dates_and_positions(*args, )
        data = expanded_data[_np.ix_(pos, columns)]
        num_columns = data.shape[1]
        new = Series(num_columns=num_columns, data_type=self.data_type)
        new.set_data(dates, data)
        return new

    def _resolve_dates_and_positions(
        self,
        dates: Dates,
        columns: ColumnsRequestType = None,
        /,
    ) -> tuple[Iterable[_dates.Dater], Iterable[int], Iterable[int], _np.ndarray]:
        """
        """
        dates = self._resolve_dates(dates, )
        dates = [ t for t in dates ]
        columns = self._resolve_columns(columns)
        if not dates:
            dates = ()
            pos = ()
            data = self._create_periods_of_missing_values(num_rows=0, )[:, columns]
            return dates, pos, columns, data
        #
        base_date = self.start_date or _builtin_min(dates, )
        pos, add_before, add_after = _get_date_positions(dates, base_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after, )
        if not isinstance(pos, Iterable):
            pos = (pos, )
        return dates, pos, columns, data

    def get_data(
        self,
        dates = ...,
        *args,
    ) -> _np.ndarray:
        dates, pos, columns, expanded_data = self._resolve_dates_and_positions(dates, *args, )
        data = expanded_data[_np.ix_(pos, columns)]
        return data

    def get_data_column(
        self,
        dates: Dates,
        column: Number | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        column = column if column and column<self.data.shape[1] else 0
        return self.get_data(dates, column, )

    def get_data_from_to(
        self,
        from_to,
        *args,
    ) -> _np.ndarray:
        """
        """
        _, pos, columns, expanded_data \
            = self._resolve_dates_and_positions(from_to, *args, )
        from_pos, to_pos = pos[0], pos[-1]+1
        return expanded_data[from_pos:to_pos, columns]

    def get_data_column_from_to(
        self,
        from_to: Iterable[_dates.Dater],
        column: int | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        column = column if column and column < self.data.shape[1] else 0
        return self.get_data_from_to(from_to, column, )

    def iter_data_columns_from_to(
        self,
        from_to: Iterable[_dates.Dater],
        /,
    ) -> Iterator[_np.ndarray]:
        """
        Iterate over the columns and yield 1-D arrays for the given time span
        """
        data_from_to = self.get_data_from_to(from_to, )
        yield from data_from_to.T

    def extract_columns(
        self,
        columns,
        /,
    ) -> None:
        if not isinstance(columns, Iterable):
            columns = (columns, )
        else:
            columns = tuple(c for c in columns)
        self.data = self.data[:, columns]
        self.column_titles = [ self.column_titles[c] for c in columns ]

    def set_start_date(
        self,
        new_start_date: _dates.Dater,
        /,
    ) -> Self:
        self.start_date = new_start_date
        return self

    def _resolve_dates(self, dates, ):
        """
        """
        if dates is None:
            dates = []
        if isinstance(dates, slice) and dates == slice(None, ):
            dates = ...
        if dates is ... and self.start_date is not None:
            dates = _dates.Ranger(None, None, )
        if dates is ... and self.start_date is None:
            dates = _dates.EmptyRanger()
        if hasattr(dates, "needs_resolve") and dates.needs_resolve:
            dates = dates.resolve(self, )
        return tuple(
            d.resolve(self) if hasattr(d, "needs_resolve") and d.needs_resolve else d
            for d in dates
        )

    def _resolve_columns(
        self,
        columns: ColumnsRequestType,
        /,
    ) -> Iterable[int]:
        """
        Resolve column request to an iterable of integers
        """
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

    def shift(
        self,
        by: int | str = -1,
        /,
    ) -> None:
        match by:
            case "yoy":
                self._shift_by_number(-self.frequency.value, )
            case "boy" | "soy":
                self._shift_to_soy()
            case "eopy":
                self._shift_to_eopy()
            case "tty":
                self._shift_to_tty()
            case _:
                self._shift_by_number(by, )

    def _shift_by_number(self, by: int, ) -> None:
        """
        Shift (lag, lead) start date by a number of periods
        """
        self.start_date = (
            self.start_date - by
            if self.start_date else self.start_date
        )

    def _shift_to_soy(self, ) -> None:
        new_data = self.get_data(
            t.create_soy()
            for t in self.range
        )
        self._replace_data(new_data, )

    def _shift_to_tty(self, ) -> None:
        new_data = self.get_data(
            t.create_tty()
            for t in self.range
        )
        self._replace_data(new_data, )

    def _shift_to_eopy(self, ) -> Self:
        new_data = self.get_data(
            t.create_eopy()
            for t in self.range
        )
        self._replace_data(new_data, )

    def __getitem__(self, index):
        """
        Create a new time series based on date retrieved by self[dates] or self[dates, columns]
        """
        if isinstance(index, int):
            new = self.copy()
            new.shift(index)
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
            return self.copy()
        encompassing_range = _dates.get_encompassing_range(self, *args, )
        new_data = self.get_data(encompassing_range, )
        add_data = (
            x.get_data(encompassing_range, )
            if hasattr(x, "get_data")
            else _create_data_column_from_number(x, encompassing_range, self.data_type)
            for x in args
        )
        new_data = _np.hstack((new_data, *add_data))
        new = Series(num_columns=new_data.shape[1], )
        new.set_data(encompassing_range, new_data, )
        new.trim()
        return new

    def clip(
        self,
        new_start_date: _dates.Dater | None,
        new_end_date: _dates.Dater | None,
    ) -> None:
        if new_start_date is None or new_start_date < self.start_date:
            new_start_date = self.start_date
        if new_end_date is None or new_end_date > self.end_date:
            new_end_date = self.end_date
        if new_start_date == self.start_date and new_end_date == self.end_date:
            return
        self.start_date = new_start_date
        self.data = self.get_data(_dates.Ranger(new_start_date, new_end_date))

    def is_empty(self, ) -> bool:
        return not self.data.size

    def empty(
        self,
        *,
        num_columns: int | None = None
    ) -> None:
        num_columns = num_columns if num_columns is not None else self.num_columns
        self.data = _np.empty((0, num_columns), dtype=self.data_type)

    def overlay_by_range(
        self,
        other: Self,
        /,
    ) -> Self:
        self.set_data(other.range, other.data, )
        self.trim()

    def overlay(
        self,
        other: Self,
        /,
        method = "by_range",
    ) -> Self:
        _broadcast_columns_if_needed(self, other, )
        self._LAY_METHOD_RESOLUTION[method]["overlay"](self, other, )

    def underlay_by_range(
        self,
        other: Self,
        /,
    ) -> Self:
        """
        """
        new_self = other.copy()
        new_self.overlay(self, )
        self._shallow_copy_data(new_self, )

    def underlay(
        self,
        other: Self,
        /,
        method = "by_range",
    ) -> Self:
        _broadcast_columns_if_needed(self, other, )
        self._LAY_METHOD_RESOLUTION[method]["underlay"](self, other, )

    _LAY_METHOD_RESOLUTION = {
        "by_range": {"overlay": overlay_by_range, "underlay": underlay_by_range},
    }

    def redate(
        self,
        new_date: _dates.Dater,
        old_date: _dates.Dater | None = None,
        /,
    ) -> None:
        """
        """
        self.start_date = new_date \
            if old_date is None \
            else new_date - (old_date - self.start_date)

    def _shallow_copy_data(
        self,
        other: Self,
        /,
    ) -> None:
        """
        """
        for n in ("start_date", "data", "data_type", ):
            setattr(self, n, getattr(other, n, ))

    @property
    def column_titles(self, ) -> tuple[str, ...]:
        """
        """
        return self._column_titles

    @column_titles.setter
    def column_titles(
        self,
        column_titles: Iterable[str] | str,
    ) -> None:
        """
        """
        if not column_titles:
            self._column_titles = ("", ) * self.num_columns
            return
        if isinstance(column_titles, str):
            column_titles = (column_titles, )
        if len(self._column_titles) == 1:
            self._column_titles = tuple(self._column_titles) * self.num_columns
        if len(column_titles) != self.num_columns:
            raise _wrongdoings.IrisPieError(
                "Number of column titles must match number of data columns"
            )

    def _reset_column_titles_if_needed(
        self,
        /,
    ) -> None:
        """
        """
        if not self._column_titles or self.num_columns != len(self._column_titles):
            self._column_titles = ("", ) * self.num_columns

    def __or__(self, other):
        """
        Implement the | operator
        """
        return self.hstack(other)

    def __lshift__(self, other):
        """
        Implement the << operator
        """
        return self.copy().overlay_by_range(other, )

    def __rshift__(self, other):
        """
        Implement the >> operator
        """
        return other.copy().overlay_by_range(self, )

    def trim(self):
        if self.data.size == 0:
            self.reset()
            return self
        num_leading = _get_num_leading_missing_rows(self.data, self._test_missing_period)
        if num_leading == self.data.shape[0]:
            self.reset()
            return self
        self._reset_column_titles_if_needed()
        num_trailing = _get_num_leading_missing_rows(self.data[::-1], self._test_missing_period)
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading or None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to), ...]
        if slice_from:
            self.start_date += int(slice_from)
        return self

    def _create_expanded_data(self, add_before, add_after):
        return _np.pad(
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
        output = self.copy()
        output.data = -output.data
        return output

    def __pos__(self):
        """
        +self
        """
        return self.copy()

    def __add__(self, other) -> Self|_np.ndarray:
        """
        self + other
        """
        # arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, _op.add, )

    def __radd__(self, other):
        """
        other + self
        """
        return self.apply(lambda data: data.__radd__(other))

    def __mul__(self, other) -> Self|_np.ndarray:
        """
        self + other
        """
        # arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, _op.mul, )

    def __rmul__(self, other):
        """
        other + self
        """
        return self.apply(lambda data: data.__rmul__(other))

    def __sub__(self, other):
        """
        self - other
        """
        # arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, _op.sub, )

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
        # arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, _op.pow, )

    def __rpow__(self, other):
        """
        other ** self
        """
        return self.apply(lambda data: data.__rpow__(other))

    def __truediv__(self, other):
        """
        self - other
        """
        # arg = other.data if isinstance(other, type(self)) else other
        return self._binop(other, _op.truediv, )

    def __rtruediv__(self, other):
        """
        other - self
        """
        return self.apply(lambda data: data.__rtruediv__(other))

    def __floordiv__(self, other):
        """
        self // other
        """
        return self._binop(other, _op.floordiv, )

    def __rfloordiv__(self, other):
        """
        other // self
        """
        return self.apply(lambda data: data.__rfloordiv__(other))

    def __mod__(self, other):
        """
        self % other
        """
        return self._binop(other, _op.mod, )

    def __rmod__(self, other, /, ):
        """
        other % self
        """
        return self.apply(lambda data: data.__rmod__(other))

    def apply(self, func, /, *args, **kwargs, ):
        new_data = func(self.data, *args, **kwargs, )
        axis = kwargs.get("axis", )
        if new_data.shape == self.data.shape:
            new = self.copy()
            new._replace_data(new_data, )
            return new
        elif (axis is None or axis == 1) and new_data.shape == (self.data.shape[0], ):
            new = self.copy()
            new.data = new_data.reshape(self.data.shape[0], 1, )
            new.trim()
            return new
        elif axis is None and new_data.shape == ():
            return new_data
        elif axis == 0 and new_data.shape == (self.data.shape[1], ):
            return new_data
        else:
            raise _wrongdoings.IrisPieError(
                "Function applied on a time series resulted"
                " in a data array with an unexpected shape"
            )

    def _binop(self, other, func, /, new=None, ):
        if not isinstance(other, type(self)):
            return self.apply(lambda data: func(data, other))
        # FIXME: empty encompassing range
        encompassing_range = _dates.get_encompassing_range(self, other)
        new_start_date = encompassing_range[0]
        self_data = self.get_data(encompassing_range, )
        other_data = other.get_data(encompassing_range, )
        new_data = func(self_data, other_data)
        new = Series(num_columns=new_data.shape[1], ) if new is None else new
        new._replace_start_date_and_values(new_start_date, new_data, )
        return new

    def _broadcast_columns(self, num_columns, /, ) -> None:
        """
        """
        if self.data.shape[1] == num_columns:
            return
        if self.data.shape[1] == 1:
            self.data = _np.repeat(self.data, num_columns, axis=1, )
            return
        raise _wrongdoings.IrisPieError("Cannot broadcast columns")

    def _replace_data(
        self,
        new_values,
        /,
    ) -> None:
        """
        """
        self.data = new_values
        self.trim()

    def _replace_start_date_and_values(
        self,
        new_start_date,
        new_values,
        /,
    ) -> None:
        """
        """
        self.start_date = new_start_date
        self._replace_data(new_values, )

    def __iter__(self, ):
        """
        Default iterator is line by line, yielding a tuple of (date, values)
        """
        return zip(self.range, self.data)

    def iter_data_variants_from_to(self, from_to, /, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        return _iterators.exhaust_then_last(self.iter_data_columns_from_to(from_to, ))

    for n in FUNCTION_ADAPTATIONS:
        exec(f"def _{n}_(self, *args, **kwargs, ): return self.apply(_np.{n}, *args, **kwargs, )")

    #]


Series.from_dates_and_data = _wrongdoings.obsolete(Series.from_dates_and_values)
Series.from_start_date_and_data = _wrongdoings.obsolete(Series.from_start_date_and_values)


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


def _create_data_column_from_number(
    number: Number,
    range: _dates.Ranger,
    data_type: type,
    /,
) -> _np.ndarray:
    return _np.full((len(range), 1), number, dtype=data_type)


def _conform_data(data, /, data_type, ) -> _np.ndarray:
    """
    """
    #[
    # Tuple means columns
    if isinstance(data, tuple, ):
        return _np.hstack(tuple(
            _conform_data(d, data_type=data_type, ) for d in data
        ))
    #
    if not isinstance(data, _np.ndarray, ):
        return _np.array(data, dtype=data_type, ndmin=2, ).T
    #
    return _np.array(data, dtype=data_type, ndmin=2, )
    #]


for n in FUNCTION_ADAPTATIONS_NUMPY:
    exec(
        f"@_ft.singledispatch"
        f"\n"
        f"def {n}(x, *args, **kwargs): "
        f"return _np.{n}(x, *args, **kwargs)"
    )
    exec(
        f"@{n}.register(Series, )"
        f"\n"
        f"def _(x, *args, **kwargs): "
        f"return x._{n}_(*args, **kwargs)"
    )


for n in FUNCTION_ADAPTATIONS_BUILTINS:
    exec(
        f"_builtin_{n} = {n}"
    )
    exec(
        f"@_ft.singledispatch"
        f"\n"
        f"def {n}(x, *args, **kwargs): "
        f"return _builtin_{n}(x, *args, **kwargs)"
    )
    exec(
        f"@{n}.register(Series, )"
        f"\n"
        f"def {n}(x, *args, **kwargs): "
        f"return x._{n}_(*args, **kwargs)"
    )


def _from_dates_and_values(
    self,
    dates: Iterable[_dates.Dater],
    values: _np.ndarray | Iterable,
    **kwargs,
) -> None:
    """
    """
    #[
    values = _conform_data(values, data_type=self.data_type, )
    num_columns = values.shape[1] if hasattr(values, "shape") else 1
    self.empty(num_columns=num_columns, )
    self.set_data(dates, values, )
    #]


def _from_dates_and_func(
    self,
    dates: Iterable[_dates.Dater],
    func: Callable,
    **kwargs,
) -> Self:
    """
    Create a new time series from dates and a function
    """
    #[
    dates = tuple(dates)
    data = [
        [func() for j in range(self.num_columns)]
        for i in range(len(dates))
    ]
    data = _np.array(data, dtype=self.data_type)
    self.set_data(dates, data)
    #]


def _from_start_date_and_values(
    self,
    start_date: _dates.Dater,
    values: _np.ndarray | Iterable,
    **kwargs,
) -> None:
    """
    """
    #[
    values = _conform_data(values, data_type=self.data_type, )
    self.start_date = start_date
    self.data = values
    self.trim()
    #]


def _broadcast_columns_if_needed(
    self: Series,
    other: Series,
    /,
) -> tuple[Series, Series]:
    """
    """
    #[
    if self.num_columns == other.num_columns:
        return self, other
    #
    if self.num_columns == 1:
        return self._broadcast_columns(other.num_coluns, ), other
    #
    if other.num_columns == 1:
        return self, other._broadcast_columns(self.num_columns, )
    #
    raise _wrongdoings.IrisPieError("Cannot broadcast time series columns")
    #]


def _invalid_constructor(
    self,
    **kwargs,
) -> NoReturn:
    raise _wrongdoings.IrisPieError("Invalid Series object constructor")


_SERIES_FACTORY = {
    (False, False, False, False): lambda self, **kwargs: None,
    (True, False, True, False): _from_start_date_and_values,
    (False, True, True, False): _from_dates_and_values,
    (False, True, False, True): _from_dates_and_func,
}


def shift(
    self,
    by: int | str = -1,
) -> _series.Series:
    """
    """
    #[
    new = self.copy()
    new.shift(by)
    return new
    #]


for n in ("underlay", "overlay", "redate", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )

