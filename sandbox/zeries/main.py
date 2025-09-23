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
import itertools as _it
import functools as _ft
import operator as _op
import copy as _cp

from ..conveniences import copies as _copies
from ..conveniences import iterators as _iterators
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants

from . import _diffcums
from . import _fillings
from . import _movings
from . import _conversions
from . import _hp
from . import _x13

from . import _plotly
from . import _views
from . import _functionalize
from . import _invariants
from . import _variants

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

DatesType: TypeAlias = _dates.Dater | Iterable[_dates.Dater] | EllipsisType | None
VariantsType: TypeAlias = int | Iterable[int] | slice | None

def _trim_decorate(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        for v in self._variants:
            v.trim()
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
    _has_invariant.HasInvariantMixin,
    _has_variants.HasVariantsMixin,
    _views.ViewMixin,
    _copies.CopyMixin,
):
    """
    Time series objects
    """
    #[

    # __slots__ = (
    #     "_invariant", "_variants",
    # )

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
        num_variants: int = 1,
        start_date: _dates.Dater | None = None,
        dates: Iterable[_dates.Dater] | None = None,
        values: Any | None = None,
        func: Callable | None = None,
        data_type: type = _np.float64,
        description: str | None = None,
        custom_data: dict[str, Any] | None = None,
        skeleton: bool = False,
    ) -> None:
        """
        """
        self._invariant = None
        self._variants = []
        if skeleton:
            return
        self._invariant = _invariants.Invariant(
            description=description,
            custom_data=custom_data,
        )
        constructor_test = tuple(x is not None for x in (start_date, dates, values, func))
        variant_constructor = _ft.partial(
            _variants.SERIES_VARIANT_FACTORY.get(constructor_test, _invalid_constructor, ),
            start_date=start_date, dates=dates, func=func, data_type=data_type,
        )
        self._variants = [
            variant_constructor(values=values, )
            for vid, values in zip(range(num_variants, ), _has_variants.iter_variants(values), )
        ]

    @classmethod
    def skeleton(klass, /, ) -> None:
        """
        """
        return klass(skeleton=True, )

    def reset(self, /, ) -> None:
        """
        """
        for v in self._variants:
            v.reset()

    @property
    def shape(self, /, ) -> tuple[int, int]:
        return (self.num_periods, self.num_variants, )

    @property
    def num_periods(self, /, ) -> int:
        start_date = self.start_date
        return self.end_date - start_date + 1 if start_date else 0

    @property
    def start_date(self, /, ) -> _dates.Dater | None:
        try:
            return _builtin_min( v.start_date for v in self._variants )
        except ValueError:
            return None

    @property
    def end_date(self, /, ) -> _dates.Dater | None:
        try:
            return _builtin_max( v.end_date for v in self._variants )
        except ValueError:
            return None

    @property
    def span(self, /, ):
        return (
            _dates.Ranger(self.start_date, self.end_date, )
            if self.start_date else _dates.EmptyRanger
        )

    range = span

    @property
    def dates(self, /, ) -> tuple[_dates.Dater, ...]:
        return tuple(self.range, )

    @property
    def frequency(self):
        start_date = self.start_date
        return (
            start_date.frequency
            if start_date else _dates.Frequency.UNKNOWN
        )

    @classmethod
    def from_dates_and_values(
        klass,
        dates: DatesType,
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

    def set_data(
        self,
        dates: DatesType,
        values,
        vids: VariantsType = None,
        /,
    ) -> None:
        dates = self._resolve_dates(dates, )
        vids = self._resolve_vids(vids, )
        iter_values = _has_variants.iter_variants(values, )
        for vid, values in zip(vids, iter_values):
            self._variants[vid].set_data(dates, values, )

    def _resolve_dates_and_positions(
        self,
        dates: DatesType,
        vids: VariantsType = None,
        /,
    ) -> tuple[Iterable[_dates.Dater], Iterable[int], Iterable[int], _np.ndarray]:
        """
        """
        dates = self._resolve_dates(dates, )
        dates = [ t for t in dates ]
        vids = self._resolve_vids(vids)
        if not dates:
            dates = ()
            pos = ()
            data = self._create_periods_of_missing_values(num_rows=0, )[:, vids]
            return dates, pos, vids, data
        #
        base_date = self.start_date or _builtin_min(dates, )
        pos, add_before, add_after = _get_date_positions(dates, base_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after, )
        if not isinstance(pos, Iterable):
            pos = (pos, )
        return dates, pos, vids, data

    def get_data(
        self,
        dates: DatesType = ...,
        vids: VariantsType = None,
    ) -> _np.ndarray:
        """
        """
        dates = self._resolve_dates(dates, )
        vids = self._resolve_vids(vids, )
        return _concatenate_data(
            self._variants[i].get_data(dates, )
            for i in vids
        )

    def get_data_from_to(self, from_to, /, ) -> _np.ndarray:
        """
        """
        from_to = self._resolve_dates(from_to, )
        return _concatenate_data( v.get_data_from_to(from_to, ) for v in self._variants )

    def get_data_variant_from_to(
        self,
        from_to: Iterable[_dates.Dater],
        vid: int | None = None,
        /,
    ) -> _np.ndarray:
        """
        """
        from_to = self._resolve_dates(from_to, )
        vid = vid if vid is not None and variant < self.num_variants else self.num_variant
        return self._variants[vid].get_data_from_to(from_to, ).reshape(-1, 1)

    def _resolve_dates(self, dates, /, ):
        """
        """
        resolution_context = _dates.ResolutionContext(self.start_date, self.end_date, )
        #
        if dates is None:
            dates = ()
        if isinstance(dates, slice) and dates == slice(None, ):
            dates = ...
        if dates is ... and resolution_context.start_date is not None:
            dates = _dates.Ranger(None, None, )
        if dates is ... and start_date is None:
            dates = ()
        if hasattr(dates, "needs_resolve") and dates.needs_resolve:
            dates = dates.resolve(resolution_context, )
        #
        return tuple(
            d.resolve(resolution_context, ) if hasattr(d, "needs_resolve") and d.needs_resolve else d
            for d in dates
        )

    def _resolve_from_to(self, from_to, /, ) -> tuple[_dates.Dater, _dates.Dater, ]:
        """
        """
        from_to = self._resolve_dates(from_to, )
        return from_to[0], from_to[-1]

    def _resolve_vids(
        self,
        variants: VariantsType,
        /,
    ) -> list[int]:
        """
        Resolve variant request to an iterable of integers
        """
        if variants is None or variants is Ellipsis:
            variants = slice(None, )
        if isinstance(variants, slice, ):
            variants = range(*variants.indices(self.num_variants))
        if not isinstance(variants, Iterable, ):
            variants = [variants, ]
        return list(variants)

    def __call__(self, *args, ):
        """
        Get data self[dates] or self[dates, variants]
        """
        return self.get_data(*args, )

    def shift(
        self,
        by: int | str = -1,
        /,
    ) -> None:
        """
        """
        match by:
            case "yoy":
                func = _ft.partial(_variants.Variant.shift_by, by=-self.frequency.value, )
            case "boy" | "soy":
                func = _variants.Variant.shift_soy
            case "eopy":
                func = _variants.Variant.shift_eopy
            case "tty":
                func = _variants.Variant.shift_tty
            case _:
                func = _ft.partial(_variants.Variant.shift_by, by=by, )
        for v in self._variants:
            func(v)

    def __getitem__(self, index):
        """
        Create a new time series based on date retrieved by self[dates] or self[dates, variants]
        """
        new = self.copy()
        if isinstance(index, int):
            new.shift(index)
            return new
        if not isinstance(index, tuple):
            index = (index, None, )
        dates, vids, *_ = index
        dates = self._resolve_dates(dates, )
        vids = self._resolve_vids(vids, )
        new._variants = [ new._variants[i].copy() for i in vids ]
        for v in new._variants:
            v.keep_dates(dates, )
        return new

    def __setitem__(self, index, data, ):
        """
        Set data self[dates] = ... or self[dates, variants] = ...
        """
        if not isinstance(index, tuple):
            index = (index, None, )
        dates, vids, *_ = index
        return self.set_data(dates, data, vids, )

    def hstack(
        self,
        *args,
    ) -> Self:
        """
        """
        new = self.copy()
        for other in args:
            start_date = self.start_date
            for v in other._variants:
                _check_freq_consistency(start_date, v.start_date, )
                new._variants.append(v)
        return new

    def clip(
        self,
        from_to: Iterable[_dates.Dater],
    ) -> None:
        """
        """
        from_to = self._resolve_from_to(from_to, )
        for v in self._variants:
            v.clip(from_to, )

    def is_empty(self, ) -> bool:
        return all( v.is_empty() for v in self._variants )

    def empty(self, *args, **kwargs, ) -> None:
        """
        """
        for v in self._variants:
            v.empty(*args, **kwargs, )

    def overlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        self.broadcast_variants(other, )
        for self_variant, other_variant in self._iterate_extract_self_other_variants(other, ):
            self_variant.overlay(other_variant, **kwargs, )

    def underlay(
        self,
        other: Self,
        **kwargs,
    ) -> Self:
        """
        """
        new = other.copy()
        new.overlay(self, **kwargs, )
        self._variants = new._variants

    def redate(self, *args, **kwargs, ) -> None:
        """
        """
        for v in self._variants:
            v.redate(*args, **kwargs, )

    def __or__(self, other, /, ):
        """
        Implement the | operator
        """
        return self.hstack(other, )

    def __neg__(self, /, ):
        """
        -self
        """
        new = self.copy()
        for v in new._variants:
            v.data = -v.data
        return new

    def __pos__(self):
        """
        +self
        """
        return self.copy()

    def __add__(self, other, /, ) -> Self:
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
        return self.apply(lambda data: data.__rmod__(other, ), )

    def apply(self, func, *args, **kwargs, ):
        """
        """
        new_values_from_variants = [
            v.apply(func, *args, **kwargs, )
            for v in self._variants
        ]
        test = [
            isinstance(v, _variants.Variant)
            for v in new_values_from_variants
        ]
        if all(test):
            new = Series()
            new._invariant = self._invariant.copy()
            new._variants = new_values_from_variants
            return new
        elif any(test):
            _wrongdoings.IrisPieCritical(
                "Inconsistent return types from applying "
                "a function to Series variants "
            )
        else:
            return new_values_from_variants

    def _binop(self, other, func, /, ):
        """
        """
        if isinstance(other, type(self), ):
            return self._binop_series(other, func, )
        else:
            def apply_func(data, /, ):
                return func(data, other, )
            return self.apply(apply_func, )

    def _binop_series(self, other, func, /, ) -> Self:
        """
        """
        _, new_start_date, new_end_date = _dates.get_encompassing_span(self, other, )
        from_to = (new_start_date, new_end_date, )
        new = Series(num_variants=0, )
        #
        def _calculate_new_values(self_variant, other_variant, /, ) -> _np.ndarray:
            self_values = self_variant.get_data_from_to(from_to, )
            other_values = other_variant.get_data_from_to(from_to, )
            new_values = func(self_values, other_values, )
            if not isinstance(new_values, _np.ndarray) \
                or new_values.shape != self_values.shape:
                _wrongdoings.IrisPieCritical(
                    "Inconsistent return types from applying "
                    "a function to Series variants "
                )
            return new_values
        #
        for self_variant, other_variant in self._iterate_extract_self_other_variants(other, ):
            new_values = (
                _calculate_new_values(self_variant, other_variant, )
                if new_start_date is not None else ()
            )
            new_variant = _variants.Variant.from_start_date_and_values(
                start_date=new_start_date,
                values=new_values,
            )
            new._variants.append(new_variant)
        #
        return new

    def _iterate_extract_self_other_variants(self, other, /, ) -> Iterator[Tuple[_variants.Variant, _variants.Variant]]:
        """
        """
        new_num_variants = _builtin_max(self.num_variants, other.num_variants, )
        iterate_max_times = zip(
            range(new_num_variants),
            self.iter_extract_variants(),
            other.iter_extract_variants(),
        )
        for _, self_variant, other_variant in iterate_max_times:
            yield self_variant, other_variant

    def __iter__(self, ):
        """
        """
        new = Series.skeleton()
        new._invariant = self._invariant
        for v in self._variants:
            new._variants = [ v, ]
            yield new

    def iter_data_variants_from_to(self, from_to, /, ) -> Iterator[_np.ndarray]:
        """
        Iterates over the data variants from the given start date to the given end date
        """
        iterator = ( v.get_data_from_to(from_to, ) for v in self._variants )
        return _iterators.exhaust_then_last(iterator, )

    for n in FUNCTION_ADAPTATIONS:
        exec(f"def _{n}_(self, *args, **kwargs, ): return self.apply(_np.{n}, *args, **kwargs, )")

    #]


Series.from_dates_and_data = _wrongdoings.obsolete(Series.from_dates_and_values)
Series.from_start_date_and_data = _wrongdoings.obsolete(Series.from_start_date_and_values)


def hstack(first, *args, ) -> Self:
    return first.hstack(*args, )


def _create_data_variant_from_number(
    number: Real,
    span: _dates.Ranger,
    data_type: type,
    /,
) -> _np.ndarray:
    return _np.full((len(span), 1), number, dtype=data_type, )


def _conform_data(data, /, data_type, ) -> _np.ndarray:
    """
    """
    #[
    # Tuple means variants
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
    num_variants = values.shape[1] if hasattr(values, "shape") else 1
    self.empty(num_variants=num_variants, )
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


def _invalid_constructor(*args, **kwargs, ) -> NoReturn:
    raise _wrongdoings.IrisPieError("Invalid Series object constructor")


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


def _concatenate_data(data_iterator: Iterable[_np.ndarray], /, ) -> _np.ndarray:
    """
    """
    return _np.hstack(tuple(d.reshape(-1, 1, ) for d in data_iterator), )


def _check_freq_consistency(
    one: _dates.Dater | None,
    other: _dates.Dater | None,
) -> None | NoReturn:
    """
    """
    #[
    if one is None or other is None:
        return
    if one.frequency == other.frequency:
        return
    raise _wrongdoings.IrisPieCritical(
        "Cannot concatenate time series with different frequencies",
        f"{one.frequency}",
        f"{other.frequency}",
    )
    #]

