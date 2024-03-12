"""
"""


#[
from __future__ import annotations

import json as _js
import copy as _co
import types as _ty
import numpy as _np
import re as _re
import operator as _op
import functools as _ft
import itertools as _it
import os as _os
from typing import (Self, TypeAlias, Literal, Sequence, Protocol, Any, )
from collections.abc import (Iterable, Iterator, Callable, )
from numbers import (Number, )

from ..conveniences import views as _views
from ..conveniences import descriptions as _descriptions
from ..conveniences import iterators as _iterators
from ..series import main as _series
from ..dataslates import main as _dataslates
from .. import quantities as _quantities
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import pages as _pages

from . import _imports as _imports
from . import _exports as _exports
from . import _merge as _merge
from . import _dotters as _dotters
from . import _fred as _fred
from . import _views as _views
#]


__all__ = (
    "Databox",
    "Databank",
)


SourceNames: TypeAlias = Iterable[str] | str | Callable[[str], bool] | None
TargetNames: TypeAlias = Iterable[str] | str | Callable[[str], str] | None
InterpretRange: TypeAlias = Literal["base", "extended", ]


class SteadyDataboxableProtocol(Protocol):
    """
    """
    #[

    def get_min_max_shifts(self, *args) -> tuple(int, int): ...
    def generate_steady_items(self, *args) -> Any: ...

    #]


def _extended_range_tuple_from_base_span(
    input_span: Iterable[_dates.Dater],
    min_shift: int,
    max_shift: int,
    prepend_initial: bool,
    append_terminal: bool,
    /,
) -> tuple[_dates.Date, _dates.Date]:
    """
    """
    range_list = tuple(t for t in input_span)
    start_date, end_date = range_list[0], range_list[-1]
    start_date += min_shift if prepend_initial else 0
    end_date += max_shift if append_terminal else 0
    return start_date, end_date


@_pages.reference(
    path=("data_management", "databoxes.md", ),
    categories={
        "constructor": "Creating new databoxes",
        "property": None,
        "import_export": "Importing and exporting databoxes",
    },
)
class Databox(
    _imports.Inlay,
    _exports.ExportMixin,
    _merge.Inlay,
    _dotters.DotterMixin,
    _fred.FredMixin,
    _descriptions.DescriptionMixin,
    _views.ViewMixin,
    dict,
):
    """
......................................................................

Databoxes
==========

The `Databox` objects can be used to store and manipulate unstructured data
organized as key-value pairs, in a dictionary style; the `Databox` objects
are, indeed, subclassed from the standard `dict`. The values stored within
a databox can be of any type.

......................................................................
    """
    #[

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        """
        super().__init__(*args, **kwargs, )
        self._description = ""
        self._dotters = []

    @classmethod
    @_pages.reference(category="constructor", call_nam="Databox.empty", )
    def empty(
        klass,
        /,
    ) -> Self:
        """
......................................................................

==Create a new empty `Databox`==

......................................................................
        """
        return klass()

    @classmethod
    @_pages.reference(category="constructor", call_name="Databox.from_dict", )
    def from_dict(
        klass,
        _dict: dict,
        /,
    ) -> Self:
        """
......................................................................

==Create a `Databox` from a `dict`==

......................................................................
        """
        self = klass()
        for k, v in _dict.items():
            self[k] = v
        return self

    @classmethod
    @_pages.reference(category="constructor", call_name="Databox.from_dict", )
    def from_array(
        klass,
        array: _np.ndarray,
        names: Sequence[str],
        *,
        descriptions: Iterable[str] | None = None,
        dates: Iterable[_dates.Date] | None = None,
        start_date: _dates.Date | None = None,
        target_databox: Self | None = None,
        orientation: Literal["vertical", "horizontal", ] = "vertical",
    ) -> Self:
        """
......................................................................

==Create a `Databox` from a NumPy array==

......................................................................
        """
        array = array if orientation == "horizontal" else array.T
        series_constructor = _get_series_constructor(start_date, dates, )
        return klass._from_horizontal_array_and_constructor(
            array,
            names,
            series_constructor,
            descriptions=descriptions,
            target_databox=target_databox,
        )


    @classmethod
    def _from_horizontal_array_and_constructor(
        klass,
        array: _np.ndarray,
        names: Iterable[str],
        series_constructor: Callable,
        *,
        descriptions: Sequence[str] | None = None,
        target_databox: Self | None = None,
    ) -> Self:
        """
        """
        self = target_databox or klass()
        descriptions = (
            descriptions if descriptions is not None
            else _it.repeat("", )
        )
        for name, values, description in zip(names, array, descriptions, ):
            print(name, type(values), description, )
            # self[name] = series_constructor(values=values, description=description, )
        return self

    def iter_variants(
        self,
        /,
        *,
        item_iterator: Iterator[Any] | None = None,
        names: Iterable[str] | None = None,
    ) -> Iterator[dict]:
        """
        """
        names = names or self.keys()
        item_iterator = item_iterator or _default_item_iterator
        dict_variant_iter = {
            k: item_iterator(self[k], )
            for k in names if k in self
        }
        while True:
            yield { k: next(v, ) for k, v in dict_variant_iter.items() }

    def get_names(self, /, ) -> list[str]:
        """
        Get all names stored in a databox save for private attributes
        """
        return tuple(self.keys())

    def get_missing_names(self, names: Iterable[str], /, ) -> tuple[str]:
        """
        Get names that are not in the databox
        """
        return tuple(name for name in names if name not in self)

    def get_num_items(self, /, ) -> int:
        """
        """
        return len(self.keys())

    @property
    def num_items(self, /, ) -> int:
        return len(self.keys())

    def to_dict(self: Self) -> dict:
        """
        Convert Databox to dict
        """
        return { k: v for k, v in self.items() }

    def copy(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        strict_names: bool = False,
    ) -> Self:
        """
        """
        new_databox = _co.deepcopy(self)
        if source_names is None and target_names is None:
            return new_databox
        source_names, target_names, *_ = self._resolve_source_target_names(
            source_names, target_names, strict_names,
        )
        new_databox.rename(source_names, target_names, )
        new_databox.keep(target_names, )
        return new_databox

    def shallow(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        strict_names: bool = False,
    ) -> Self:
        """
        """
        source_names, target_names, *_ = self._resolve_source_target_names(
            source_names, target_names, strict_names,
        )
        return type(self)(
            (t, self[s])
            for s, t in zip(source_names, target_names, )
        )

    def rename(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        strict_names: bool = False,
    ) -> None:
        """
        """
        source_names, target_names, *_ = self._resolve_source_target_names(
            source_names, target_names, strict_names,
        )
        for s, t in zip(source_names, target_names, ):
            if s == t:
                continue
            self[t] = self.pop(s)

    def remove(
        self: Self,
        /,
        remove_names: SourceNames = None,
        strict_names: bool = False,
    ) -> None:
        """
        """
        if remove_names is None:
            return self
        context_names = self.get_names()
        remove_names, *_ = self._resolve_source_target_names(
            remove_names, None, strict_names,
        )
        for n in remove_names:
            del self[n]

    def keep(
        self: Self,
        /,
        keep_names: SourceNames = None,
        strict_names: bool = False,
    ) -> None:
        """
        """
        if keep_names is None:
            return self
        keep_names, _, context_names = self._resolve_source_target_names(
            keep_names, None, strict_names,
        )
        for n in context_names:
            if n in keep_names:
                continue
            del self[n]

    def apply(
        self,
        func: Callable,
        /,
        source_names: SourceNames = None,
        in_place: bool = True,
        when_fails: Literal["critical", "error", "warning", "silent", ] = "critical",
        strict_names: bool = False,
    ) -> None:
        """
        """
        source_names, *_ = self._resolve_source_target_names(
            source_names, None, strict_names,
        )
        when_fails_stream = \
            _wrongdoings.STREAM_FACTORY[when_fails] \
            (f"Error(s) when applying function to Databox items:")
        for s in source_names:
            try:
                output = func(self[s])
                if not in_place:
                    self[s] = output
            except Exception as e:
                when_fails_stream.add(f"{s}: {repr(e)}", )
        when_fails_stream._raise()

    def filter(
        self,
        /,
        name_test: Callable | None = None,
        value_test: Callable | None = None,
    ) -> Iterable[str]:
        """
        """
        names = tuple(self.get_names())
        if name_test is None and value_test is None:
            return names
        name_test = name_test if name_test else lambda x: True
        value_test = value_test if value_test else lambda x: True
        return tuple(
            name
            for name in names
            if name_test(name) and value_test(self[name], )
        )

    def get_series_names_by_frequency(
        self,
        frequency: _dates.Frequency,
    ) -> Iterable[str]:
        """
        Get all time series names with the given frequency
        """
        def _is_series_with_frequency(x):
            return isinstance(x, _series.Series) and x.frequency == frequency
        return self.filter(value_test=_is_series_with_frequency, )

    def get_span_by_frequency(
        self,
        frequency: _dates.Frequency,
    ) -> _dates.Ranger:
        """
        Get the encompassing date range for all time series with the given frequency
        """
        if frequency == _dates.Frequency.UNKNOWN:
            return _dates.EmptyRanger()
        names = self.get_series_names_by_frequency(frequency)
        if not names:
            return _dates.EmptyRanger()
        start_dates = (self[n].start_date for n in names)
        end_dates = (self[n].end_date for n in names)
        min_start_date = min(start_dates, key=_op.attrgetter("serial"), )
        max_end_date = max(end_dates, key=_op.attrgetter("serial"), )
        return _dates.Ranger(min_start_date, max_end_date, )

    def to_json(self, file_name, **kwargs):
        """
        """
        with open(file_name, "wt+") as f:
            return _js.dump(self, f, **kwargs)

    def overlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> None:
        """
        """
        self._lay(other, _series.Series.overlay, **kwargs)

    def underlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> None:
        """
        """
        self._lay(other, _series.Series.underlay, **kwargs)

    def _lay(
        self,
        other: Self,
        func: Callable,
        /,
        names: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        """"
        """
        if names is None:
            value_test = lambda x: isinstance(x, _series.Series)
            self_names = self.filter(value_test=value_test, )
            other_names = other.filter(value_test=value_test, )
            names = set(self_names).intersection(other_names)
        for n in names:
            if self[n].frequency == other[n].frequency:
                func(self[n], other[n], **kwargs, )

    def clip(
        self,
        new_start_date: _dates.Dater | None = None,
        new_end_date: _dates.Dater | None = None,
        /,
    ) -> None:
        """
        """
        if new_start_date is None and new_end_date is None:
            return
        frequency = (
            new_start_date.frequency
            if new_start_date is not None
            else new_end_date.frequency
        )
        value_test = lambda x: isinstance(x, _series.Series) and x.frequency == frequency
        names = self.filter(value_test=value_test, )
        for n in names:
            self[n].clip(new_start_date, new_end_date, )

    def prepend(
        self,
        prepending: Self,
        end_prepending: _dates.Dater,
        /,
    ) -> Self:
        """
        """
        prepending = prepending.copy()
        prepending.clip(None, end_prepending, )
        self.underlay(prepending, )

    def evaluate_expression(
        self,
        expression: str,
        /,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        """
        expression = expression.strip()
        if expression in self:
            return self[expression]
        else:
            return self.eval(expression, context, )

    def eval(
        self,
        expression: str,
        /,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        """
        expression = _reformat_eval_expression(expression, )
        globals = { k: v for k, v in self.items() }
        if context:
            globals |= context
        return eval(expression, globals, )

    def __call__(
        self,
        expression: str,
        /,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        """
        return self.eval(expression, context, )

    @classmethod
    def steady(
        klass,
        steady_databoxable: SteadyDataboxableProtocol,
        input_span: Iterable[_dates.Dater],
        /,
        deviation: bool = False,
        prepend_initial: bool = True,
        append_terminal: bool = True,
    ) -> Self:
        """
        """
        self = klass()
        min_shift, max_shift = steady_databoxable.get_min_max_shifts()
        start_date, end_date = _extended_range_tuple_from_base_span(
            input_span,
            min_shift,
            max_shift,
            prepend_initial,
            append_terminal,
        )
        items = steady_databoxable.generate_steady_items(start_date, end_date, deviation=deviation, )
        for name, value in items:
            self[name] = value
        return self

    zero = _ft.partialmethod(steady, deviation=True, )

    def minus_control(
        self,
        model,
        control: Self,
        /,
    ) -> None:
        """
        """
        MINUS_FACTORY = {
            True: lambda x, y: x / y,
            False: lambda x, y: x - y,
            None: lambda x, y: x - y,
        }
        kind = _quantities.QuantityKind.ANY_VARIABLE | _quantities.QuantityKind.ANY_SHOCK
        quantities = model.get_quantities(kind=kind, )
        qid_to_name = _quantities.create_qid_to_name(quantities, )
        qid_to_logly = _quantities.create_qid_to_logly(quantities, )
        for qid, name in qid_to_name.items():
            minus_func = MINUS_FACTORY[qid_to_logly[qid]]
            self[name] = minus_func(self[name], control[name], )

    def __or__(self, other) -> Self:
        new = _co.deepcopy(self)
        new.update(other, )
        return new

    def _resolve_source_target_names(
        self,
        /,
        source_names: SourceNames,
        target_names: TargetNames,
        strict_names: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """
        """
        context_names = self.get_names()
        if source_names is None:
            source_names = context_names
        if isinstance(source_names, str):
            source_names = tuple(source_names)
        if callable(source_names):
            func = source_names
            source_names = tuple(n for n in context_names if func(n))
        if target_names is None:
            target_names = source_names
        if isinstance(target_names, str):
            target_names = tuple(target_names)
        if callable(target_names):
            func = target_names
            target_names = tuple(func(n) for n in source_names)
        if not strict_names:
            source_target_pairs = tuple((s, t) for s, t in zip(source_names, target_names, ) if s in context_names)
            source_names, target_names = zip(*source_target_pairs, ) if source_target_pairs else ((), (), )
        return source_names, target_names, context_names

    #]


Databank = Databox


def _apply_to_item(
    func: Callable,
    source: Any,
    target: Any,
    /,
) -> None:
    """
    Apply function to source, capture result in target
    """
    target = source
    return func(target)


def _reformat_eval_expression(expression: str, ) -> str:
    """
    """
    #[
    if "=" in expression:
        lhs, *rhs = expression.split(expression, "=", )
        rhs = "=".join(rhs, )
        expression = "({lhs})-({rhs})"
    return expression
    #]


def _default_item_iterator(value: Any, /, ) -> Iterator[Any]:
    """
    """
    #[
    is_value_iterable = (
        isinstance(value, Iterable)
        and not isinstance(value, str)
        and not isinstance(value, bytes)
    )
    value = value if is_value_iterable else [value, ]
    yield from _iterators.exhaust_then_last(value, None, )
    #]


def _get_series_constructor(
    start_date: _dates.Dater | None = None,
    dates: Iterable[_dates.Dater] | None = None,
    /,
) -> Callable:
    """
    """
    #[
    if start_date is not None:
        return _ft.partial(
            _series.Series.from_start_date_and_values,
            start_date=start_date,
        )
    elif dates is not None:
        return _ft.partial(
            _series.Series.from_dates_and_values,
            dates=dates,
        )
    #]

