"""
Frequency conversion of time series
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
import numpy as _np
import functools as _ft
import operator as _op

from .. import dates as _dates
#]


class Mixin:
    """
    """
    #[
    def aggregate(
        self,
        new_freq: _dates.Frequency,
        /,
        method: str = "mean",
        remove_missing: bool = False,
        select: list[int] | None = None,
    ) -> Self:
        """
        Aggregate time series to a lower frequency
        """
        method_func = _AGGREGATION_METHOD_RESOLUTION[method]
        #
        if new_freq == self.frequency:
            return self
        if new_freq > self.frequency or new_freq is _dates.Frequency.UNKNOWN or self.frequency is _dates.Frequency.UNKNOWN:
            raise ValueError(f"Cannot aggregate from {self.frequency} frequency to {new_freq} frequency")
        #
        new_dater_class = _dates.DATER_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
        #
        aggregate_within_data_func = _ft.partial(
            _aggregate_within_data,
            select,
            remove_missing,
            method_func,
        )
        #
        if self.frequency.is_regular:
            aggregate_func = _aggregate_regular_to_regular
        elif self.frequency is _dates.Frequency.DAILY:
            aggregate_func = _aggregate_daily_to_regular
        #
        new_start_date, new_data = aggregate_func(self, new_dater_class, aggregate_within_data_func)
        return type(self).from_start_date_and_values(new_start_date, new_data, )

    def disaggregate(
        self,
        new_freq: _dates.Frequency,
        /,
        method: str = "flat",
    ) -> Self:
        """
        Disaggregate time series to a higher frequency
        """
        method_func = _DISAGGREGATION_METHOD_RESOLUTION[method]
        #
        if new_freq == self.frequency:
            return self
        if new_freq < self.frequency or new_freq is _dates.Frequency.UNKNOWN or self.frequency is _dates.Frequency.UNKNOWN:
            raise ValueError(f"Cannot aggregate from {self.frequency} frequency to {new_freq} frequency")
        #
        new_dater_class = _dates.DATER_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
        new_start_date, new_data = method_func(self, new_dater_class)
        return type(self).from_start_date_and_values(new_start_date, new_data, )
    #]


def _aggregate_daily_to_regular(
    self,
    new_dater_class: type,
    aggregate_within_data_func: Callable,
) -> Self:
    """
    """
    #[
    start_date = self.start_date.create_soy()
    end_date = self.end_date.create_eoy()
    all_self_data = self.get_data(_dates.Ranger(start_date, end_date))
    start_year = self.start_date.get_year()
    end_year = self.end_date.get_year()
    new_start_date = new_dater_class.from_year_period(start_year, 1)
    new_end_date = new_dater_class.from_year_period(end_year, "end")
    get_slice_func = lambda t: slice(
        t.to_daily(position="start") - start_date,
        t.to_daily(position="end") - start_date + 1,
    )
    new_data = tuple(
        tuple(
            aggregate_within_data_func(column[get_slice_func(t)])
            for column in all_self_data.T
        )
        for t in _dates.Ranger(new_start_date, new_end_date)
    )
    #
    new_data = _np.array(new_data, dtype=self.data_type)
    return new_start_date, new_data
    #]


def _aggregate_regular_to_regular(
    self,
    new_dater_class: type,
    aggregate_within_data_func: Callable,
    /,
) -> tuple[Dater, _np.ndarray]:
    """
    """
    #[
    start_year = self.start_date.get_year()
    start_date = self.start_date.create_soy()
    end_date = self.end_date.create_eoy()
    new_start_date = new_dater_class.from_year_period(start_year, 1)
    new_freq = new_dater_class.frequency
    self_data = self.get_data(_dates.Ranger(start_date, end_date))
    factor = self.frequency.value // new_freq
    #
    # Loop over columns, create within-period data, and aggregate
    new_data_transposed = tuple(
        tuple(
            aggregate_within_data_func(within_data)
            for within_data in self_data_column.reshape((-1, factor))
        )
        for self_data_column in self_data.T
    )
    #
    new_data_transposed = _np.array(new_data_transposed, dtype=self.data_type)
    return new_start_date, new_data_transposed.T
    #]


def _disaggregate_flat(
    self,
    new_dater_class: type,
    /,
) -> tuple[Dater, _np.ndarray]:
    """
    """
    #[
    new_freq = new_dater_class.frequency
    new_start_date = new_dater_class.from_ymd(*self.start_date.to_ymd(position="start"))
    new_end_date = new_dater_class.from_ymd(*self.end_date.to_ymd(position="end"))
    new_num_periods = new_end_date - new_start_date + 1
    #
    get_slice_func = lambda t: slice(
        new_dater_class.from_ymd(*t.to_ymd(position="start")) - new_start_date,
        new_dater_class.from_ymd(*t.to_ymd(position="end")) - new_start_date + 1,
    )
    #
    new_data = _np.full((new_num_periods, self.num_columns), _np.nan, dtype=self.data_type)
    for t in _dates.Ranger(self.start_date, self.end_date):
        new_data[get_slice_func(t), :] = self.get_data(t)
    return new_start_date, new_data
    #]


def _aggregate_within_data(
    select: list[int] | None,
    remove_missing: bool,
    method_func: Callable,
    within_data: _np.ndarray,
    /,
) -> _np.ndarray:
    """
    """
    #[
    if select is not None:
        within_data = within_data[select]
    if remove_missing:
        within_data = within_data[~_np.isnan(within_data)]
    return method_func(within_data) if within_data.size > 0 else _np.nan
    #]


_AGGREGATION_METHOD_RESOLUTION = {
    "mean": _np.mean,
    "sum": _np.sum,
    "first": _op.itemgetter(0),
    "last": _op.itemgetter(-1),
    "min": _np.min,
    "max": _np.max,
}


_DISAGGREGATION_METHOD_RESOLUTION = {
    "flat": _disaggregate_flat,
}

