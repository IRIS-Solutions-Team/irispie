"""
Main time series class definition
"""


#[
from __future__ import annotations

from numbers import Real
from collections.abc import (Iterable, Callable, )
from typing import (Self, Any, TypeAlias, NoReturn, )
from types import (EllipsisType, )
import numpy as _np
import scipy as _sp
import itertools as _it
import functools as _ft
import operator as _op
import copy as _cp

from ..conveniences import descriptions as _descriptions
from ..conveniences import copies as _copies
from ..conveniences import iterators as _iterators
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants
from .. import pages as _pages

from . import _diffs_cums
from . import _filling
from . import _moving
from . import _conversions
from . import _hp
from . import _x13

from . import _indexing
from . import _plotly
from . import _views
from . import _functionalize

from ._diffs_cums import __all__ as _diffs_cums__all__
from ._diffs_cums import *

from ._filling import __all__ as _fillings__all__
from ._filling import *

from ._hp import __all__ as _hp__all__
from ._hp import *

from ._moving import __all__ as _moving__all__
from ._moving import *

from ._conversions import __all__ as _conversions__all__
from ._conversions import *
#]


_ELEMENTWISE_FUNCTIONS = {
    "log": _np.log,
    "exp": _np.exp,
    "sqrt": _np.sqrt,
    "abs": _np.abs,
    "sign": _np.sign,
    "sin": _np.sin,
    "cos": _np.cos,
    "tan": _np.tan,
    "round": _np.round,
    "logistic": _sp.special.expit,
    "normal_cdf": _sp.stats.norm.cdf,
    "normal_pdf": _sp.stats.norm.pdf,
}


# FIXME
FUNCTION_ADAPTATIONS_ELEMENTWISE = tuple(_ELEMENTWISE_FUNCTIONS.keys())
FUNCTION_ADAPTATIONS_NUMPY_APPLY = () # ("maximum", "minimum", "mean", "median", "abs", )
FUNCTION_ADAPTATIONS_BUILTINS = ("max", "min", )
FUNCTION_ADAPTATIONS = tuple(set(
    FUNCTION_ADAPTATIONS_ELEMENTWISE
    # + FUNCTION_ADAPTATIONS_NUMPY
    # + FUNCTION_ADAPTATIONS_BUILTINS
))


__all__ = (
    ("Series", "shift", )
    + _conversions__all__
    + _diffs_cums__all__
    + _fillings__all__
    + _hp__all__
    + _moving__all__
    + FUNCTION_ADAPTATIONS
)

Dates: TypeAlias = _dates.Dater | Iterable[_dates.Dater] | _dates.Ranger | EllipsisType | None
VariantsRequestType: TypeAlias = int | Iterable[int] | slice | None


def _get_date_positions(dates, base, num_periods, /, ):
    pos = tuple(_dates.date_index(dates, base))
    min_pos = _builtin_min((x for x in pos if x is not None), default=0)
    max_pos = _builtin_max((x for x in pos if x is not None), default=0)
    add_before = _builtin_max(-min_pos, 0)
    add_after = _builtin_max(max_pos - num_periods + 1, 0)
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


@_pages.reference(
    path=("data_management", "time_series.md", ),
    categories={
        "constructor": "Creating new time series",
        "conversion": "Converting time series frequency",
        "property": None,
    },
)
class Series(
    _indexing.Inlay,
    _conversions.Inlay,
    _diffs_cums.Inlay,
    _filling.Inlay,
    _hp.Inlay,
    _moving.Inlay,
    _x13.Inlay,
    _plotly.Inlay,

    _descriptions.DescriptionMixin,
    _views.ViewMixin,
    _copies.CopyMixin,
):
    """
················································································

Time series
============

The `Series` objects represent numerical time series, organized as rows of
observations stored in NumPy arrays and'
[date](dates.md)
stamped. A `Series` object can hold multiple
variants of the data, stored as mutliple columns.

················································································
    """
    #[

    __slots__ = (
        "start_date",
        "data",
        "data_type",
        "metadata",
        "_description", 
    )

    _numeric_format: str = "15g"
    _short_str_format: str = ">15"
    _date_str_format: str = ">12"
    _missing = _np.nan
    _missing_str: str = "⋅"
    _test_missing_period = staticmethod(lambda x: _np.all(_np.isnan(x)))

    @_pages.reference(category="constructor", call_name="Series", )
    def __init__(
        self,
        /,
        *,
        num_variants: int = 1,
        data_type: type = _np.float64,
        description: str = "",
        start_date: _dates.Dater | None = None,
        dates: Iterable[_dates.Dater] | None = None,
        frequency: _dates.Frequency | None = None,
        values: Any | None = None,
        func: Callable | None = None,
    ) -> None:
        """
················································································

==Create a new `Series` object==

```
self = Series(
    start_date=start_date,
    values=values,
)
```

```
self = Series(
    dates=dates,
    values=values,
)
```

### Input arguments ###


???+ input "start_date"

    The date of the first value in the `values`.

???+ input "values"

    Time series observations, supplied either as a tuple of values, or a
    NumPy array.

### Returns ###

???+ returns "self"

    The newly created `Series` object.

················································································
        """
        self.start_date = None
        self.data_type = data_type
        self.data = _np.full((0, num_variants), self._missing, dtype=self.data_type)
        self.metadata = {}
        self._description = description
        test = (x is not None for x in (start_date, dates, values, func))
        _SERIES_FACTORY.get(tuple(test), _invalid_constructor)(self, start_date=start_date, dates=dates, frequency=frequency, values=values, func=func, )

    def reset(self, /, ) -> None:
        self.__init__(
            num_variants=self.num_variants,
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
    @_pages.reference(category="property", )
    def shape(self, /, ) -> tuple[int, int]:
        """==Shape of time series data=="""
        return self.data.shape

    @property
    @_pages.reference(category="property", )
    def num_periods(self, /, ) -> int:
        """==Number of periods from the first to the last observation=="""
        return self.data.shape[0]

    @property
    @_pages.reference(category="property", )
    def num_variants(self, /, ) -> int:
        """==Number of variants (columns) within the `Series` object=="""
        return self.data.shape[1]

    @property
    def is_singleton(self, /, ) -> bool:
        """
        True for time series with only one variant
        """
        return _has_variants.is_singleton(self.num_variants, )

    @property
    def span(self, ):
        return _dates.Ranger(self.start_date, self.end_date, ) if self.start_date else ()

    range = span

    @property
    def from_to(self, ):
        return self.start_date, self.end_date

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
        variants: VariantsRequestType = None,
        /,
    ) -> None:
        dates = self._resolve_dates(dates)
        data = (
            _reshape_numpy_array(data, )
            if isinstance(data, _np.ndarray) else data
        )
        vids = self._resolve_variants(variants, )
        if not self.start_date:
            self.start_date = next(iter(dates), None, )
            self.data = self._create_periods_of_missing_values(num_rows=1, )
        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0], )
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before
        if hasattr(data, "get_data"):
            data = data.get_data(dates)
        #
        data_variants = _has_variants.iter_variants(data, )
        for c, d in zip(vids, data_variants, ):
            self.data[pos, c] = d

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
    ) -> tuple[Iterable[_dates.Dater], Iterable[int], Iterable[int], _np.ndarray]:
        """
        """
        dates = self._resolve_dates(dates, )
        dates = [ t for t in dates ]
        variants = self._resolve_variants(variants)
        if not dates:
            dates = ()
            pos = ()
            data = self._create_periods_of_missing_values(num_rows=0, )[:, variants]
            return dates, pos, variants, data
        #
        base_date = self.start_date or _builtin_min(dates, )
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

    def get_data(
        self,
        dates = ...,
        *args,
    ) -> _np.ndarray:
        dates, pos, variants, expanded_data = self._resolve_dates_and_positions(dates, *args, )
        data = expanded_data[_np.ix_(pos, variants)]
        return data

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

    def get_data_from_to(
        self,
        from_to,
        *args,
    ) -> _np.ndarray:
        """
        """
        _, pos, variants, expanded_data \
            = self._resolve_dates_and_positions(from_to, *args, )
        from_pos, to_pos = pos[0], pos[-1]+1
        return expanded_data[from_pos:to_pos, variants]

    def get_data_variant_from_to(
        self,
        from_to: Iterable[_dates.Dater],
        variant: int | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        variant = variant if variant and variant < self.data.shape[1] else 0
        return self.get_data_from_to(from_to, variant, )

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
            # dates = []
            dates = ...
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

    def hstack(self, *args):
        if not args:
            return self.copy()
        self_args = (self, *args, )
        if all(i.is_empty() for i in self_args):
            num_variants = sum(i.num_variants for i in self_args)
            return Series(num_variants=num_variants, )
        encompassing_span, *from_to = _dates.get_encompassing_span(self, *args, )
        new_data = self.get_data_from_to(from_to, )
        add_data = (
            x.get_data_from_to(from_to, )
            if hasattr(x, "get_data")
            else _create_data_variant_from_number(x, encompassing_span, self.data_type)
            for x in args
        )
        new_data = _np.hstack((new_data, *add_data))
        new = Series(num_variants=new_data.shape[1], )
        new.set_data(encompassing_span, new_data, )
        new.trim()
        return new

    def clip(
        self,
        /,
        new_start_date: _dates.Dater | None,
        new_end_date: _dates.Dater | None,
    ) -> None:
        if new_start_date is None or new_start_date < self.start_date:
            new_start_date = self.start_date
        if new_end_date is None or new_end_date > self.end_date:
            new_end_date = self.end_date
        if new_start_date == self.start_date and new_end_date == self.end_date:
            return
        self.data = self.get_data_from_to((new_start_date, new_end_date, ), )
        self.start_date = new_start_date

    def is_empty(self, ) -> bool:
        return not self.data.size

    def empty(
        self,
        *,
        num_variants: int | None = None
    ) -> None:
        num_variants = num_variants if num_variants is not None else self.num_variants
        self.data = _np.empty((0, num_variants), dtype=self.data_type)

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
        _broadcast_variants_if_needed(self, other, )
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
        _broadcast_variants_if_needed(self, other, )
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

    def __or__(self, other):
        """
        Implement the | operator
        """
        return self.hstack(other)

    def trim(self):
        if self.data.size == 0:
            self.reset()
            return self
        num_leading = _get_num_leading_missing_rows(self.data, self._test_missing_period)
        if num_leading == self.data.shape[0]:
            self.reset()
            return self
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
            raise Exception("Time series data being assigned must preserve the number of variants")

    def __bool__(self, /, ):
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
        _, *from_to = _dates.get_encompassing_span(self, other)
        self_data = self.get_data_from_to(from_to, )
        other_data = other.get_data_from_to(from_to, )
        new_data = func(self_data, other_data)
        new = Series(num_variants=new_data.shape[1], ) if new is None else new
        new._replace_start_date_and_values(from_to[0], new_data, )
        return new

    def _broadcast_variants(self, num_variants, /, ) -> None:
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

    def iter_variants(self, /, ) -> Iterator[Self]:
        """
        """
        for data in self.iter_data_variants_from_to(..., ):
            new = self.copy()
            new._replace_data(data, )
            yield new

    def iter_own_data_variants_from_to(self, from_to, /, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        data_from_to = self.data if from_to == ... else self.get_data_from_to(from_to, )
        return iter(data_from_to.T, )

    def iter_data_variants_from_to(self, from_to, /, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        return _iterators.exhaust_then_last(self.iter_own_data_variants_from_to(from_to, ), )

    def logistic(self, /, ) -> Self:
        """
        """
        self.data = 1 / (1 + _np.exp(-self.data))

    for n in FUNCTION_ADAPTATIONS_ELEMENTWISE:
        exec(f"def {n}(self, *args, **kwargs, ): self.data = _ELEMENTWISE_FUNCTIONS['{n}'](self.data, *args, **kwargs, )")

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


def _create_data_variant_from_number(
    number: Real,
    range: _dates.Ranger,
    data_type: type,
    /,
) -> _np.ndarray:
    return _np.full((len(range), 1), number, dtype=data_type)


for n in FUNCTION_ADAPTATIONS_ELEMENTWISE:
    exec(
        f"@_ft.singledispatch\n"
        f"def {n}(x, *args, **kwargs):\n"
        f"    return _ELEMENTWISE_FUNCTIONS['{n}'](x, *args, **kwargs)\n"
    )
    exec(
        f"@{n}.register(Series, )\n"
        f"def _(x, *args, **kwargs):\n"
        f"    new = x.copy()\n"
        f"    new.{n}(*args, **kwargs)\n"
        f"    return new\n"
    )


for n in FUNCTION_ADAPTATIONS_BUILTINS:
    exec(
        f"_builtin_{n} = {n}"
    )
#     exec(
#         f"@_ft.singledispatch"
#         f"\n"
#         f"def {n}(x, *args, **kwargs): "
#         f"return _builtin_{n}(x, *args, **kwargs)"
#     )
#     exec(
#         f"@{n}.register(Series, )"
#         f"\n"
#         f"def {n}(x, *args, **kwargs): "
#         f"return x._{n}_(*args, **kwargs)"
#     )


def _from_dates_and_values(
    self,
    dates: Iterable[_dates.Dater],
    values: _np.ndarray | Iterable,
    frequency: _dates.Frequency | None = None,
    **kwargs,
) -> None:
    """
    """
    #[
    # values = _has_variants.iter_variants(values, )
    # num_variants = values.shape[1] if hasattr(values, "shape") else 1
    # self.empty(num_variants=num_variants, )
    dates = tuple(( _dates.ensure_dater(d, frequency=frequency, ) for d in dates ))
    self.set_data(dates, values, )
    #]


def _from_dates_and_func(
    self,
    dates: Iterable[_dates.Dater],
    func: Callable,
    frequency: _dates.Frequency | None = None,
    **kwargs,
) -> Self:
    """
    Create a new time series from dates and a function
    """
    #[
    dates = tuple(( _dates.ensure_dater(d, frequency=frequency, ) for d in dates ))
    data = [
        [func() for j in range(self.num_variants)]
        for i in range(len(dates))
    ]
    data = _np.array(data, dtype=self.data_type)
    self.set_data(dates, data)
    #]


def _from_start_date_and_values(
    self,
    start_date: _dates.Dater,
    values: _np.ndarray | Iterable,
    frequency: _dates.Frequency | None = None,
    **kwargs,
) -> None:
    """
    """
    #[
    self.start_date = _dates.ensure_dater(start_date, frequency=frequency, )
    if isinstance(values, _np.ndarray):
        values = _reshape_numpy_array(values, )
    else:
        values = _has_variants.iter_variants(values, )
        values = _np.column_stack(tuple(
            v for v, _ in zip(values, range(self.num_variants), )
        ))
    values = values.astype(self.data_type, )
    self.data = values
    self.trim()
    #]


def _reshape_numpy_array(values: _np.ndarray, /, ) -> _np.ndarray:
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
    /,
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

