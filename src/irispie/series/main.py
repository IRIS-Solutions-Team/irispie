"""
Main time series class definition
"""


#[

from __future__ import annotations

from numbers import Real
from collections.abc import Iterable, Callable
from typing import Self, Any, TypeAlias, NoReturn, Literal
from types import EllipsisType
import numpy as _np
import operator as _op
import documark as _dm

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from ..conveniences import iterators as _iterators
from ..dates import (Period, Span, Frequency, EmptyRanger, )
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants

from ._functionalize import FUNC_STRING

from . import _indexing
from . import _plotly
from . import _views
from . import _temporal
from . import _filling
from . import _conversions
from . import _hp
from . import _extrapolate
from . import _x13
from . import _moving
from . import _elementwise
from . import _statistics

from ._categories import CATEGORIES

#]


#
# The following functions can return either a new time series object if applied
# across variants (axis=1), or a scalar or a list of values if applied across
# time (axis=0)
#


__all__ = (
    "Series",
    "shift",
)


Dates = Period | Iterable[Period] | Span | EllipsisType | None
VariantsRequestType = int | Iterable[int] | slice | None
LayMethodType = Literal["by_span", ]
ShiftType = int | Literal["yoy", "soy", "eopy", "tty", ]
AxisType = Literal[0, 1]


def _get_date_positions(dates, base, num_periods, ):
    pos = tuple(_dates.period_indexes(dates, base, ))
    min_pos = min((x for x in pos if x is not None), default=0)
    max_pos = max((x for x in pos if x is not None), default=0)
    add_before = max(-min_pos, 0)
    add_after = max(max_pos - num_periods + 1, 0)
    pos_adjusted = [
        p + add_before if p is not None else None
        for p in pos
    ]
    return pos_adjusted, add_before, add_after


@_dm.reference(
    path=("data_management", "time_series.md", ),
    categories=CATEGORIES,
)
class Series(
    _indexing.Inlay,
    _conversions.Inlay,
    _temporal.Inlay,
    _filling.Inlay,
    _hp.Inlay,
    _extrapolate.Inlay,
    _moving.Inlay,
    _x13.Inlay,
    _plotly.Inlay,
    _views.Inlay,
    _elementwise.Inlay,
    _statistics.Inlay,
    #
    _descriptions.DescriptionMixin,
    _copies.Mixin,
):
    r"""
················································································

Time series
============

Time `Series` objects represent numerical time series, organized as rows of
observations stored in [`numpy`](https://numpy.org) arrays and time stamped
using [time `Periods`](periods.md). A `Series` object can hold multiple
variants of the data, stored as mutliple columns.

················································································
    """
    #[

    __slots__ = (
        "start",
        "data",
        "data_type",
        "metadata",
        "__description__",
    )

    _numeric_format: str = "15g"
    _short_str_format: str = ">15"
    _date_str_format: str = ">12"
    _missing = _np.nan
    _missing_str: str = "⋅"
    _test_missing_period = staticmethod(lambda x: _np.all(_np.isnan(x)))

    @_dm.reference(
        category="constructor",
        call_name="Series",
    )
    def __init__(
        self,
        /,
        *,
        num_variants: int = 1,
        data_type: type = _np.float64,
        description: str = "",
        start: Period | None = None,
        start_date: Period | None = None,
        periods: Iterable[Period] | None = None,
        dates: Iterable[Period] | None = None,
        frequency: Frequency | None = None,
        values: Any | None = None,
        func: Callable | None = None,
        populate: bool = True,
    ) -> None:
        """
················································································

==Create a new `Series` object==

```
self = Series(
    start=start,
    values=values,
)
```

```
self = Series(
    periods=periods,
    values=values,
)
```

```
self = Series(
    periods=periods,
    func=func,
)
```


### Input arguments ###

???+ input "start"
    The time [`Period`](periods.md) of the first value in the `values`.

???+ input "periods"
    An iterable of time [`Periods`](periods.md) that will be used to time stamp
    the `values`. The iterable can be e.g. a tuple, a list, a time
    [`Span`](spans.md), or a single time [`Period`](periods.md).

???+ input "values"
    Time series values, supplied either as a single values, a tuple of values,
    or a NumPy array.

???+ input "func"
    A function that will be used to populate the time series; the function
    should not take any input arguments, and should return a single (scalar)
    numerical value; the function will called once for each period and each
    variant.


### Returns ###

???+ returns "None"
    This method modifies `self` in-place and does not return a value.


················································································
        """
        self.start = None
        self.data_type = data_type
        self.data = _np.full((0, num_variants), _np.nan, dtype=self.data_type, )
        self.metadata = {}
        self.__description__ = description
        #
        start = start_date if start is None else start
        periods = dates if periods is None else periods
        #
        if populate:
            test = (x is not None for x in (start, periods, values, func))
            populator = _SERIES_POPULATOR.get(tuple(test), _invalid_constructor)
            populator(
                self,
                start=start,
                periods=periods,
                frequency=frequency,
                values=values,
                func=func,
            )

    @classmethod
    def as_empty(
        klass,
        num_variants: int = 1,
        data_type: type = _np.float64,
        description: str = "",
    ) -> Self:
        r"""
        """
        return klass(
            num_variants=num_variants,
            data_type=data_type,
            description=description,
            populate=False,
        )

    @classmethod
    def _guaranteed(
        klass,
        start: Period,
        values: _np.ndarray,
        description: str = "",
    ) -> Self:
        """
        """
        self = klass()
        self.start = start
        self.data = values
        self.__description__ = description
        self.trim()
        return self

    def reset(self, ) -> None:
        self.__init__(
            num_variants=self.num_variants,
            data_type=self.data_type,
        )

    def _create_periods_of_missing_values(self, num_rows=0) -> _np.ndarray:
        """
        """
        return _np.full(
            (num_rows, self.shape[1], ),
            _np.nan,
            dtype=self.data_type
        )

    @property
    @_dm.reference(category="property", )
    def shape(self, ) -> tuple[int, int]:
        """==Shape of time series data=="""
        return self.data.shape

    @property
    @_dm.reference(category="property", )
    def num_periods(self, ) -> int:
        """==Number of periods from the first to the last observation=="""
        return self.data.shape[0]

    @property
    @_dm.reference(category="property", )
    def num_variants(self, ) -> int:
        """==Number of variants (columns) within the `Series` object=="""
        return self.data.shape[1]

    @property
    def is_singleton(self, ) -> bool:
        """
        True for time series with only one variant
        """
        return _has_variants.is_singleton(self.num_variants, )

    @property
    @_dm.reference(category="property", )
    def span(self, ):
        """==Time span of the time series=="""
        return Span(self.start, self.end, ) if self.start else ()

    range = span

    @property
    @_dm.reference(category="property", )
    def from_until(self, ):
        """==Two-tuple with the start date and end date of the time series=="""
        return self.start, self.end

    @property
    @_dm.reference(category="property", )
    def periods(self, ) -> tuple[Period, ...]:
        """==N-tuple with the periods from the start period to the end period of the time series=="""
        return tuple(self.range, )

    dates = periods

    @property
    @_dm.reference(category="property", call_name="start", )
    def _start(self):
        """==Start date of the time series=="""
        raise NotImplementedError

    @property
    def start_period(self, ):
        return self.start

    start_date = start_period

    @property
    @_dm.reference(category="property", )
    def end(self):
        """==End period of the time series=="""
        return (
            self.start + self.data.shape[0] - 1
            if self.start else None
        )

    end_period = end
    end_date = end

    @property
    @_dm.reference(category="property", )
    def frequency(self, ):
        """==Date frequency of the time series=="""
        return (
            self.start.frequency
            if self.start is not None
            else Frequency.UNKNOWN
        )

    @property
    @_dm.reference(category="property", )
    def is_empty(self, ) -> bool:
        """==True if the time series is empty=="""
        return not self.data.size

    @property
    @_dm.reference(category="property", )
    def has_missing(self, ):
        """==True if the time series is non-empty and contains in-sample missing values=="""
        return bool((not self.is_empty) and _np.isnan(self.data).any())

    def any_missing(self, *args, ) -> bool:
        """
        """
        return self._func_missing(_np.any, *args, )

    def all_missing(self, *args, ) -> bool:
        """
        """
        return self._func_missing(_np.all, *args, )

    def count_missing(self, *args, ) -> int:
        """
        """
        return self._func_missing(_np.count_nonzero, *args, )

    def _func_missing(self, func, *args, ) -> bool:
        """
        """
        data = self.get_data(*args, )
        return bool(func(_np.isnan(data)))

    def set_data(
        self,
        dates: Dates,
        data: Any | Series,
        variants: VariantsRequestType = None,
        /,
    ) -> None:
        """
        """
        dates = self.resolve_periods(dates, )
        is_data_ndarray = isinstance(data, _np.ndarray)
        is_empty_data = (
            data is None
            or (is_data_ndarray and data.size == 0)
        )
        if not dates and is_empty_data:
            return
        data = (
            _reshape_numpy_array(data, )
            if is_data_ndarray else data
        )
        vids = self._resolve_variants(variants, )
        if not self.start:
            self.start = next(iter(dates), None, )
            self.data = self._create_periods_of_missing_values(num_rows=1, )
        pos, add_before, add_after = _get_date_positions(dates, self.start, self.shape[0], )
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start -= add_before
        if hasattr(data, "get_data"):
            data = data.get_data(dates)
        #
        data_variants = _has_variants.iter_variants(data, )
        for c, d in zip(vids, data_variants, ):
            self.data[pos, c] = d
        self.trim()

    def _get_data_and_recreate(
        self,
        *args,
    ) -> _np.ndarray:
        """
        """
        dates, pos, variants, expanded_data = self._resolve_dates_and_positions(*args, )
        data = expanded_data[_np.ix_(pos, variants)]
        num_variants = data.shape[1]
        new = Series(num_variants=num_variants, data_type=self.data_type)
        new.set_data(dates, data)
        return new

    def _resolve_dates_and_positions(
        self,
        dates: Dates,
        variants: VariantsRequestType = None,
        /,
    ) -> tuple[Iterable[Period], Iterable[int], Iterable[int], _np.ndarray]:
        """
        """
        dates = self.resolve_periods(dates, )
        variants = self._resolve_variants(variants)
        if not dates:
            dates = ()
            pos = ()
            data = self._create_periods_of_missing_values(num_rows=0, )[:, variants]
            return dates, pos, variants, data
        #
        base_date = self.start or min(dates, )
        pos, add_before, add_after = _get_date_positions(dates, base_date, self.shape[0], )
        data = self._create_expanded_data(add_before, add_after, )
        if not isinstance(pos, Iterable):
            pos = (pos, )
        return dates, pos, variants, data

    def alter_num_variants(
        self,
        new_num: int,
        /,
    ) -> Self:
        """
        Alter (expand, shrink) the number of variants in this time series object
        """
        if new_num < self.num_variants:
            self.shrink_num_variants(new_num, )
        elif new_num > self.num_variants:
            self.expand_num_variants(new_num, )

    def expand_num_variants(
        self,
        new_num: int,
        /,
    ) -> None:
        """
        """
        add = new_num - self.num_variants
        if add < 0:
            raise ValueError("Use shrink_num_variants to shrink the number of variants")
        data = _np.tile(self.data[:, (-1, )], (1, add, ))
        self.data = _np.hstack((self.data, data, ), )

    def shrink_num_variants(
        self,
        new_num: int,
        /,
    ) -> None:
        """
        """
        remove = self.num_variants - new_num
        if remove < 0:
            raise ValueError("Use expand_num_variants to expand the number of variants")
        self.data = self.data[:, :-remove]

    def get_values(
        self,
        *args,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> list[tuple[Real, ...]] | tuple[Real, ...]:
        """
        """
        data = self.get_data(*args, **kwargs, )
        values = [ tuple(data_column.tolist()) for data_column in data.T ]
        return _has_variants.unpack_singleton(
            values,
            unpack_singleton=unpack_singleton,
        )

    def get_data(self, *args, **kwargs, ) -> _np.ndarray:
        return self.get_data_and_periods(*args, **kwargs, )[0]

    def get_data_and_periods(
        self,
        dates: Iterable[Period] | None | EllipsisType = ...,
        *args,
    ) -> _np.ndarray:
        periods, pos, variants, expanded_values = self._resolve_dates_and_positions(dates, *args, )
        return expanded_values[_np.ix_(pos, variants)], periods

    def get_data_variant(
        self,
        dates: Dates,
        variant: Real | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        variant = variant if variant and variant<self.data.shape[1] else 0
        return self.get_data(dates, variant, )

    def get_data_from_until(
        self,
        from_until,
        *args,
    ) -> _np.ndarray:
        """
        """
        _, pos, variants, expanded_data \
            = self._resolve_dates_and_positions(from_until, *args, )
        from_pos, to_pos = pos[0], pos[-1]+1
        return expanded_data[from_pos:to_pos, variants]

    def get_data_variant_from_until(
        self,
        from_until: Iterable[Period],
        variant: int | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        variant = variant if variant and variant < self.data.shape[1] else 0
        return self.get_data_from_until(from_until, variant, )

    def extract_variants(
        self,
        variants,
        /,
    ) -> None:
        if not isinstance(variants, Iterable):
            variants = (variants, )
        else:
            variants = tuple(c for c in variants)
        self.data = self.data[:, variants]

    def set_start(
        self,
        new_start: Period,
        /,
    ) -> Self:
        self.start = new_start
        return self

    def resolve_periods(self, periods, ) -> tuple[Period]:
        """
        """
        if periods is None:
            # periods = []
            periods = ...
        if isinstance(periods, slice) and periods == slice(None, ):
            periods = ...
        if periods is ... and self.start is not None:
            periods = Span(None, None, )
        if periods is ... and self.start is None:
            periods = EmptyRanger()
        if hasattr(periods, "needs_resolve") and periods.needs_resolve:
            periods = periods.resolve(self, )
        return tuple(
            d.resolve(self) if hasattr(d, "needs_resolve") and d.needs_resolve else d
            for d in periods
        )

    def _resolve_variants(
        self,
        variants: VariantsRequestType,
        /,
    ) -> Iterable[int]:
        """
        Resolve variant request to an iterable of integers
        """
        if variants is None or variants is Ellipsis:
            variants = slice(None, )
        if isinstance(variants, slice):
            variants = range(*variants.indices(self.num_variants))
        if not isinstance(variants, Iterable):
            variants = (variants, )
        return variants

    def shift(
        self,
        by: ShiftType = -1,
        **kwargs,
    ) -> None:
        r"""
................................................................................

==Shift the time series start date==

Shift the start date of the time series by a number of periods or to a specific
date.

    self.shift(by=-1, )

### Input arguments ###

???+ input "self"
    The current time series object that will be shifted.

???+ input "by"
    The number of periods to shift the observations by. If `by` is a string,
    the observations are manipulated as follows:

    * `"yoy"`: Shift all observations by one year back.

    * `"soy"`: Shift to the start of the year.

    * `"eopy"`: Shift to the end of the previous year.

................................................................................
        """
        if isinstance(by, int):
            self._shift_by_number(by, **kwargs, )
        else:
            method_name = f"_shift_{by}"
            getattr(self, method_name)(**kwargs, )

    def _shift_by_number(self, by: int, **kwargs, ) -> None:
        r"""
        Shift (lag, lead) start date by a number of periods
        """
        if self.start is None:
            return
        self.start -= by

    def _shift_yoy(self, /, **kwargs, ) -> None:
        r"""
        Shift the start date by one year back
        """
        self._shift_by_number(-self.frequency.value, )

    def _shift_soy(self, /, **kwargs, ) -> None:
        r"""
        Replace each observation by the start of the year observation
        """
        new_data = self.get_data(
            t.create_soy()
            for t in self.range
        )
        self._replace_data(new_data, )

    def _shift_eopy(self, /, **kwargs, ) -> Self:
        r"""
        Replace each observation by the end of the previous year observation
        """
        new_data = self.get_data(
            t.create_eopy()
            for t in self.range
        )
        self._replace_data(new_data, )

    def _shift_tty(
        self,
        by: Any | None = None,
        *,
        neutral_value: int | None = None,
        **kwargs,
    ) -> None:
        r"""
        Replace each observation by the start of the year observation except for
        the start-of-year periods which are filled with a neutral value.
        """
        zipped = tuple((t, t.create_tty(), ) for t in self.span)
        neutral_periods = tuple(t for t, tty in zipped if tty is None)
        periods = (t for t, tty in zipped if tty is not None )
        dty_periods = (tty for _, tty in zipped if tty is not None)
        dty_values = self.get_data(dty_periods, )
        self.set_data(periods, dty_values, )
        self.set_data(neutral_periods, neutral_value, )

    def hstack(self, *args, ):
        """
        """
        if not args:
            return self.copy()
        self_args = (self, *args, )
        if all(i.is_empty for i in self_args):
            num_variants = sum(i.num_variants for i in self_args)
            return Series(num_variants=num_variants, )
        encompassing_span, *from_until = _dates.get_encompassing_span(self, *args, )
        new_data = self.get_data_from_until(from_until, )
        add_data = (
            x.get_data_from_until(from_until, )
            if hasattr(x, "get_data")
            else _create_data_variant_from_number(x, encompassing_span, self.data_type)
            for x in args
        )
        new_data = _np.hstack((new_data, *add_data))
        new = Series(num_variants=new_data.shape[1], )
        new.set_data(encompassing_span, new_data, )
        return new

    @_dm.reference(category="homogenizing", )
    def clip(
        self,
        /,
        new_start: Period | None,
        new_end: Period | None,
    ) -> None:
        r"""
················································································

==Clip time series to a new start and end period==

    self.clip(new_start, new_end)


### Input arguments ###


???+ input "new_start"
    The new start period for the `self` time series; if `None`, the current
    start period is kept.

???+ input "new_end"
    The new end period for the `self` time series; if `None`, the current
    end period is kept.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


················································································
        """
        if new_start is None or new_start < self.start:
            new_start = self.start
        if new_end is None or new_end > self.end:
            new_end = self.end
        if new_start == self.start and new_end == self.end:
            return
        self.data = self.get_data_from_until((new_start, new_end, ), )
        self.start = new_start

    def empty(
        self,
        *,
        num_variants: int | None = None
    ) -> None:
        num_variants = num_variants if num_variants is not None else self.num_variants
        self.data = _np.empty((0, num_variants), dtype=self.data_type)

    @_dm.no_reference
    def overlay_by_span(
        self,
        other: Self,
        /,
    ) -> None:
        self.set_data(other.range, other.data, )
        self.trim()

    @_dm.reference(category="multiple", )
    def overlay(
        self,
        other: Self,
        /,
        method: LayMethodType = "by_span",
    ) -> None:
        r"""
................................................................................

==Overlay another time series values onto the current time series==

Overlay the values of another time series onto the current time series on the
entire span of the other time series, i.e. from the start to the end period
regardless of missing in-sample values.

    self.overlay(
        other,
        method="by_span",
    )

### Input arguments ###

???+ input "self"
    The current time series object.

???+ input "other"
    The time series object whose values will be overlaid onto the current time
    series.

???+ input "method"
    The method to use for overlaying the values. The default (and currently the
    only available) method is `"by_span"`.

### Returns ###

This method modifies `self` in place and returns `None`.

### Details ###

???+ abstract "Algorithm"

    The resulting time series is determined the following way:

    1. The span of the resulting series starts at the earliest start period of the two
    series and ends at the latest end period of the two series.

    2. The observations from the `self` (current) time series used to fill the
    resulting time span.

    3. Within the span of the `other` time series (from the first available
    observation to the last available observation), the observations from this
    `other` time series are superimposed on the resulting time series, including any
    in-sample missing observations.

................................................................................
"""
        _broadcast_variants_if_needed(self, other, )
        getattr(type(self), "overlay_" + method)(self, other, )

    @_dm.no_reference
    def underlay_by_span(
        self,
        other: Self,
        /,
    ) -> None:
        r"""
        """
        new_self = other.copy()
        new_self.overlay(self, )
        self._shallow_copy_data(new_self, )

    @_dm.reference(category="multiple", )
    def underlay(
        self,
        other: Self,
        /,
        method: LayMethodType = "by_span",
    ) -> None:
        r"""
................................................................................

==Underlay another time series values beneath the current time series==

Underlay the values of another time series beneath the current time series on
the entire span of the other time series, i.e. from the start to the end period
regardless of missing in-sample values.

    self.underlay(
        other,
        method="by_span",
    )

### Input arguments ###

???+ input "self"
    The current time series object.

???+ input "other"
    The time series object whose values will be underlaid beneath the current
    time series.

???+ input "method"
    The method to use for underlaying the values. The default (and currently the
    only available) method is `"by_span"`.

### Returns ###

This method modifies `self` in place and returns `None`.

### Details ###

???+ abstract "Algorithm"

    The resulting time series is determined the following way:

    1. The span of the resulting series starts at the earliest start period of the two
    series and ends at the latest end period of the two series.

    2. The observations from the `other` time series used to fill the
    resulting time span.

    3. Within the span of the `self` time series (from the first available
    observation to the last available observation), the observations from this
    `self` time series are superimposed on the resulting time series, including any
    in-sample missing observations.

................................................................................
        """
        _broadcast_variants_if_needed(self, other, )
        getattr(type(self), "underlay_" + method)(self, other, )

    def redate(
        self,
        new_date: Period,
        old_date: Period | None = None,
    ) -> None:
        """
        """
        if old_data is None:
            self.start = new_date
        else:
            self.start = new_date - (old_date - self.start)

    @_dm.reference(category="manipulation", )
    def replace_where(
        self,
        test: Callable,
        new_value: Real,
    ) -> None:
        """
················································································

==Replace time series values that pass a test==

```
self.replace_where(
    test,
    new_value,
)
```


### Input arguments ###


???+ input "self"
    Time series whose observations will be tested and those passing the test
    replaced.

???+ input "test"
    A function (or a Callable) that takes a numpy array and returns `True` or
    `False` for each individual value.

???+ input "new_value"
    The value to replace the observations that pass the test.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


················································································
        """
        self.data[test(self.data)] = new_value
        self.trim()

    def _shallow_copy_data(
        self,
        other: Self,
        /,
    ) -> None:
        """
        """
        for n in ("start", "data", "data_type", ):
            setattr(self, n, getattr(other, n, ))

    def __and__(self, other):
        """
        Implement the & operator as hstack
        """
        return self.hstack(other, )

    __or__ = __and__

    @_dm.reference(category="manipulation", )
    def trim(self, ) -> None:
        r"""
................................................................................

==Trim time series data==

Trim leading and trailing missing values from the time series data.

    self.trim()

### Input arguments ###

???+ input "self"
    The time series object to trim.

### Returns ###

This method modifies `self` in place and returns `None`.

................................................................................
        """
        if self.data.size == 0:
            self.reset()
            return self
        num_leading, num_trailing \
            = _get_num_leading_trailing_missing_rows(self.data, )
        if num_trailing == self.data.shape[0]:
            self.reset()
            return self
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading or None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice_from:slice_to, ...]
        if slice_from:
            self.start += int(slice_from)
        return self

    def _create_expanded_data(self, add_before, add_after):
        return _np.pad(
            self.data, ((add_before, add_after), (0, 0)),
            mode="constant", constant_values=_np.nan,
        )

    def _check_data_shape(self, data, ):
        if data.shape[1] != self.data.shape[1]:
            raise Exception("Time series data being assigned must preserve the number of variants")

    def __bool__(self, ):
        """
        """
        return self.data.size > 0

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

    def __add__(self, other, ) -> Self | _np.ndarray | Real:
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

    def __rmod__(self, other, ):
        """
        other % self
        """
        return self.apply(lambda data: data.__rmod__(other))

    def __abs__(self, *args, **kwargs, ) -> Self:
        """
        abs(self)
        """
        new = self.copy()
        new.abs(*args, **kwargs, )
        return new

    def abs(self, *args, **kwargs, ) -> None:
        r"""
        """
        self.data = _np.abs(self.data, *args, **kwargs, )

    def __round__(self, *args, **kwargs, ) -> Self:
        r"""
        """
        new = self.copy()
        new.round(*args, **kwargs, )
        return new

    def round(self, *args, **kwargs, ) -> None:
        self.data = _np.round(self.data, *args, **kwargs, )

    for n in ["gt", "lt", "ge", "le", "eq", "ne", ]:
        exec(f"def __{n}__(self, other): return self._binop(other, _op.{n}, )", )

    def apply(self, func, /, *args, **kwargs, ):
        new_data = func(self.data, *args, **kwargs, )
        axis = kwargs.get("axis", None, )
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
        _, *from_until = _dates.get_encompassing_span(self, other)
        self_data = self.get_data_from_until(from_until, )
        other_data = other.get_data_from_until(from_until, )
        new_data = func(self_data, other_data)
        new = Series(num_variants=new_data.shape[1], ) if new is None else new
        new._replace_start_and_values(from_until[0], new_data, )
        return new

    def _broadcast_variants(self, num_variants, ) -> None:
        """
        """
        if self.data.shape[1] == num_variants:
            return
        if self.data.shape[1] == 1:
            self.data = _np.repeat(self.data, num_variants, axis=1, )
            return
        raise _wrongdoings.IrisPieError("Cannot broadcast variants")

    def _replace_data(
        self,
        new_values,
        /,
    ) -> None:
        """
        """
        self.data = new_values
        self.trim()

    def _replace_start_and_values(
        self,
        new_start,
        new_values,
        /,
    ) -> None:
        """
        """
        self.start = new_start
        self._replace_data(new_values, )

    def iter_dates_values(self, unpack_singleton=True, ):
        """
        Default iterator is line by line, yielding a tuple of (date, values)
        """
        def _unpack_singleton_data_row(data_list: list[Real], ):
            return _has_variants.unpack_singleton(data_list, True, )
        def _keep_data_row(data_list: list[Real], ):
            return data_list
        data_row_func = (
            _unpack_singleton_data_row
            if self.is_singleton and unpack_singleton
            else _keep_data_row
        )
        for date, data_row in zip(self.range, self.data, ):
            yield date, data_row_func(data_row.tolist(), )

    def iter_variants(self, ) -> Iterator[Self]:
        """
        """
        for data in self.iter_data_variants_from_until(..., ):
            new = self.copy()
            new._replace_data(data, )
            yield new

    def iter_own_data_variants_from_until(self, from_until, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        if from_until == ...:
            data_from_until = self.data
        else:
            data_from_until = self.get_data_from_until(from_until, )
        return iter(data_from_until.T, )

    def iter_data_variants_from_until(self, from_until, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        return _iterators.exhaust_then_last(self.iter_own_data_variants_from_until(from_until, ), )

    def logistic(self, ) -> Self:
        """
        """
        self.data = 1 / (1 + _np.exp(-self.data))

    #]


def _get_num_leading_trailing_missing_rows(data: _np.ndarray, ):
    """
    """
    #[
    # Boolean index of rows with at least one observation
    boolex_observations = ~_np.all(_np.isnan(data, ), axis=1, )
    if _np.all(boolex_observations, ):
        # All rows have at least one observation
        num_leading = 0
        num_trailing = 0
    elif not _np.any(boolex_observations, ):
        # No row has any observation
        num_leading = 0
        num_trailing = data.shape[0]
    else:
        # Some rows have observations
        num_leading = _np.argmax(boolex_observations)
        num_trailing = _np.argmax(boolex_observations[::-1])
    return num_leading, num_trailing
    #]


def hstack(first, *args) -> Self:
    return first.hstack(*args)


def _create_data_variant_from_number(
    number: Real,
    span: Span,
    data_type: type,
    /,
) -> _np.ndarray:
    return _np.full((len(span), 1), number, dtype=data_type)


def _from_periods_and_values(
    self,
    periods: Iterable[Period] | str,
    values: _np.ndarray | Iterable,
    frequency: Frequency | None = None,
    **kwargs,
) -> None:
    """
    """
    #[
    # dates = _dates.ensure_period_tuple(dates, frequency=frequency, )
    self.set_data(periods, values, )
    #]


def _from_periods_and_func(
    self,
    periods: Iterable[Period] | str,
    func: Callable,
    frequency: Frequency | None = None,
    **kwargs,
) -> Self:
    """
    Create a new time series from dates and a function
    """
    #[
    # dates = _dates.ensure_period_tuple(dates, frequency=frequency, )
    data = [
        [func() for j in range(self.num_variants)]
        for i in range(len(periods))
    ]
    data = _np.array(data, dtype=self.data_type)
    self.set_data(periods, data)
    #]


def _from_start_and_values(
    self,
    start: Period,
    values: _np.ndarray | Iterable,
    frequency: Frequency | None = None,
    **kwargs,
) -> None:
    """
    """
    #[
    # start = _dates.ensure_period_tuple(start, frequency=frequency, )[0]
    self.start = start
    if isinstance(values, _np.ndarray):
        values = _reshape_numpy_array(values, )
    else:
        values = _has_variants.iter_variants(values, )
        values = _np.column_stack([
            v for v, _ in zip(values, range(self.num_variants), )
        ])
    values = values.astype(self.data_type, )
    self.data = values
    self.trim()
    #]


def _reshape_numpy_array(values: _np.ndarray, ) -> _np.ndarray:
    """
    """
    #[
    return (
        values.reshape(values.shape[0], -1)
        if values.ndim >= 2 else values.reshape(-1, 1)
    )
    #]


def _broadcast_variants_if_needed(
    self: Series,
    other: Series,
) -> tuple[Series, Series]:
    """
    """
    #[
    if self.num_variants == other.num_variants:
        return self, other
    #
    if self.num_variants == 1:
        return self._broadcast_variants(other.num_variants, ), other
    #
    if other.num_variants == 1:
        return self, other._broadcast_variants(self.num_variants, )
    #
    raise _wrongdoings.IrisPieError("Cannot broadcast time series variants")
    #]


def _invalid_constructor(
    self,
    **kwargs,
) -> NoReturn:
    raise _wrongdoings.IrisPieError("Invalid Series object constructor")


_SERIES_POPULATOR = {
    (False, False, False, False): lambda self, **kwargs: None,
    (True, False, True, False): _from_start_and_values,
    (False, True, True, False): _from_periods_and_values,
    (False, True, False, True): _from_periods_and_func,
}


for n in ("shift", "redate", "underlay", "overlay", ):
    code = FUNC_STRING.format(n=n, )
    exec(code, globals(), locals(), )
    __all__ += (n, )

