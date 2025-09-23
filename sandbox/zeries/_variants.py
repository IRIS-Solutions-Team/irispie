"""
Time series variants
"""


#[
from __future__ import annotations

from typing import (Self, Any, )
import numpy as _np
import copy as _co

from ..conveniences import copies as _copies
from .. import dates as _dates
#]


_builtin_min = min
_builtin_max = max


def _get_date_positions(dates, base, num_periods, /, ):
    pos = tuple(_dates.date_index(dates, base, ), )
    min_pos = _builtin_min((x for x in pos if x is not None), default=0, )
    max_pos = _builtin_max((x for x in pos if x is not None), default=0, )
    add_before = _builtin_max(-min_pos, 0, )
    add_after = _builtin_max(max_pos - (num_periods - 1), 0, )
    pos_adjusted = [
        p + add_before if p is not None else None
        for p in pos
    ]
    return pos_adjusted, add_before, add_after


class Variant:
    """
    """
    #[

    __slots__ = (
        "tag",
        "start_date",
        "data",
        "data_type",
        "trimmable",
    )

    def __init__(
        self: Self,
        *,
        data_type: type = _np.float64,
        trimmable: bool = True,
        **kwargs,
    ) -> None:
        """
        """
        self.tag: str | None = None
        self.data_type: type = data_type
        self.start_date: _dates.Dater | None = None
        self.data: _np.ndarray = _np.full((0, ), _np.nan, dtype=self.data_type, )
        self.trimmable: bool = trimmable

    @classmethod
    def from_start_date_and_values(
        klass,
        *,
        start_date: _dates.Dater,
        values: _np.ndarray | Real | tuple[Real, ...],
        trimmable: bool = True,
        **kwargs,
    ) -> None:
        """
        """
        self = klass(**kwargs, )
        self.trimmable = trimmable
        values = _conform_data(values, data_type=self.data_type, )
        self.start_date = start_date
        self.data = (
            values if self.start_date is not None
            else _np.array([], dtype=self.data_type, )
        )
        self.trim()
        return self

    @classmethod
    def from_dates_and_values(
        klass,
        *,
        dates: Iterable[_dates.Dater],
        values: _np.ndarray | Iterable,
        trimmable: bool = True,
        **kwargs,
    ) -> None:
        """
        """
        self = klass(**kwargs, )
        self.trimmable = trimmable
        values = _conform_data(values, data_type=self.data_type, )
        self.set_data(dates, values, )
        return self

    @classmethod
    def from_dates_and_func(
        klass,
        *,
        dates: Iterable[_dates.Dater],
        func: Callable,
        trimmable: bool = True,
        **kwargs,
    ) -> Self:
        """
        Create a new time series from dates and a function
        """
        self = klass(**kwargs, )
        self.trimmable = trimmable
        data = tuple(func() for i in range(len(dates)))
        data = _conform_data(data, data_type=self.data_type, )
        self.set_data(dates, data)
        return self

    @property
    def end_date(self):
        return (
            self.start_date + (self.num_periods - 1)
            if self.start_date else None
        )

    @property
    def num_periods(self):
        return self.data.size

    @property
    def span(self, /, ):
        if self.start_date is None:
            return _dates.EmptyRanger()
        else:
            return _dates.Ranger(self.start_date, self.end_date, )

    @property
    def dates(self, /, ):
        return tuple(self.span)

    @property
    def size(self):
        return self.data.size

    def copy(self, /, ) -> Self:
        """
        """
        return _co.deepcopy(self, )

    def reset(self, /, ) -> None:
        """
        """
        self.__init__(data_type=self.data_type, )

    def set_data(
        self: Self,
        resolved_dates: tuple[_dates.Dater],
        data: Any,
        /,
        trimmable: bool = True,
    ) -> None:
        """
        """
        if not self.start_date:
            self.start_date = next(iter(resolved_dates), None, )
            self.data = self._create_periods_of_missing_values(num_rows=1, )
        pos, add_before, add_after = \
                _get_date_positions(resolved_dates, self.start_date, self.num_periods, )
        self.data = self._padded_data(add_before, add_after, )
        if add_before:
            self.start_date -= add_before
        if hasattr(data, "get_data"):
            data = data.get_data(resolved_dates, )
        if isinstance(data, _np.ndarray):
            self.data[pos] = data.flatten()
        else:
            self.data[pos] = data

    def overlay(
        self,
        other: Self,
        /,
        method: Literal["by_span", "by_date", ] = "by_span",
    ) -> None:
        """
        """
        if self.start_date is None:
            self.start_date = other.start_date
            self.data = other.data.copy()
            return
        if other.start_date is None:
            return
        pos, add_before, add_after = \
            _get_date_positions(other.dates, self.start_date, self.num_periods, )
        self.data = self._padded_data(add_before, add_after, )
        other_data = other.data
        if method == "by_date":
            pos, other_data = _select_by_date(pos, other_data, )
        self.data[pos] = other_data
        self.trim()

    def get_data(
        self,
        resolved_dates: Iterable[_dates.Dater],
        *args,
    ) -> _np.ndarray:
        pos, expanded_data = self._resolve_positions_and_expand_data(resolved_dates, )
        return expanded_data[pos]

    def keep_dates(
        self,
        resolved_dates: Iterable[_dates.Dater],
        /,
    ) -> None:
        """
        """
        if self.start_date is None:
            return
        index = _dates.date_index(resolved_dates, self.start_date, )
        index = tuple(index)
        pos_nan = list(set(range(self.size)) - set(index))
        self.data[pos_nan] = _np.nan
        self.trim()

    def clip(
        self,
        from_to: tuple[_dates.Dater, _dates.Dater, ],
        /,
    ) -> None:
        """
        """
        from_to = tuple(from_to)
        new_values = self.get_data_from_to(from_to, )
        self.start_date = from_to[0]
        self.data = new_values
        self.trim()

    def apply(self, func, *args, **kwargs, ):
        """
        """
        new_data = func(self.data, *args, **kwargs, )
        if isinstance(new_data, _np.ndarray) and new_data.shape == self.data.shape:
            new = self.copy()
            new.data = new_data
            return new
        else:
            return new_data

    def is_empty(self, /, ) -> bool:
        """
        """
        return self.data.size == 0

    def empty(self, /, ) -> None:
        """
        """
        self.data = _np.empty((0, ), dtype=self.data_type, )

    def redate(
        self,
        new_date: _dates.Dater,
        old_date: _dates.Dater | None = None,
        /,
    ) -> None:
        """
        """
        old_date = (
            self.start_date if old_date is None
            else old_date
        )
        self.start_date = new_date + self.start_date - old_date

    def get_data_from_to(
        self,
        from_to: Iterable[_dates.Dater],
        *args,
    ) -> _np.ndarray:
        """
        """
        pos, expanded_data \
            = self._resolve_positions_and_expand_data(from_to, )
        from_pos, to_pos = pos[0], pos[-1]+1
        return expanded_data[from_pos:to_pos]

    def copy(self, /, ) -> Self:
        """
        """
        return _co.deepcopy(self, )

    def shift_by(self, by: int, ) -> None:
        """
        Shift (lag, lead) start date by a number of periods
        """
        self.start_date = (
            self.start_date - by if self.start_date
            else self.start_date
        )
        self.trim()

    def shift_soy(self, by=None, ) -> None:
        self.data = self.get_data(
            t.create_soy()
            for t in self.dates
        )
        self.trim()

    def shift_tty(self, by=None, ) -> None:
        self.data = self.get_data(
            t.create_tty()
            for t in self.range
        )
        self.trim()

    def shift_eopy(self, by=None, ) -> None:
        self.data = self.get_data(
            t.create_eopy()
            for t in self.range
        )
        self.trim()

    def _resolve_positions_and_expand_data(
        self,
        dates: Iterable[_dates.Dater],
        /,
    ) -> tuple[Iterable[_dates.Dater], Iterable[int], _np.ndarray]:
        """
        """
        dates = tuple(dates)
        if not dates:
            pos = ()
            data = self._create_periods_of_missing_values(num_rows=0, )
            return pos, data
        #
        base_date = self.start_date or _builtin_min(dates, )
        pos, add_before, add_after = \
            _get_date_positions(dates, base_date, self.num_periods, )
        data = self._padded_data(add_before, add_after, )
        return pos, data

    def _create_periods_of_missing_values(self, /, num_rows, ) -> _np.ndarray:
        """
        """
        return _np.full((num_rows, ), _np.nan, dtype=self.data_type, )

    def _padded_data(self, add_before, add_after, /, ):
        return _np.pad(
            self.data, ((add_before, add_after), ),
            mode="constant", constant_values=_np.nan,
        )

    def trim(self, /, ) -> None:
        """
        """
        if not self.trimmable:
            return
        if self.data.size == 0:
            self.reset()
            return self
        num_leading = _get_num_leading_missing_rows(self.data, )
        if num_leading == self.data.size:
            self.reset()
            return self
        num_trailing = _get_num_leading_missing_rows(self.data[::-1], )
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading or None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to)]
        if slice_from:
            self.start_date += int(slice_from)
        if self.data.size == 0:
            self.reset()

    #]


def _conform_data(data: Any, /, data_type, ) -> _np.ndarray:
    """
    """
    return _np.array(data, dtype=data_type, ).flatten()


def _get_num_leading_missing_rows(data, /, ):
    """
    """
    #[
    try:
        num = next(
            i for i, period_data in enumerate(data)
            if not _np.isnan(period_data)
        )
    except StopIteration:
        num = data.shape[0]
    return num
    #]


SERIES_VARIANT_FACTORY = {
    (False, False, False, False): Variant,
    (True, False, True, False): Variant.from_start_date_and_values,
    (False, True, True, False): Variant.from_dates_and_values,
    (False, True, False, True): Variant.from_dates_and_func,
}


def _select_by_date(pos, data, ):
    where_finite = _np.isfinite(data, )
    if not all(where_finite):
        data = data[where_finite]
        pos = [ p for p, w in zip(pos, where_finite, ) if w ]
    return pos, data

