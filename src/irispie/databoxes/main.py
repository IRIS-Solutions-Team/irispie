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
from typing import (Self, TypeAlias, Literal, Sequence, Protocol, Any, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )

from ..conveniences import views as _views
from ..conveniences import descriptions as _descriptions
from ..series import main as _series
from .. import quantities as _quantities
from .. import dataslates as _dataslates
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings

from . import _imports as _imports
from . import _exports as _exports
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


class EmptyDateRange(Exception):
    def __init__(self):
        super().__init__("Empty date range is not allowed in this context.")


def _extended_range_tuple_from_base_range(input_range, min_shift, max_shift):
    """
    """
    range_list = [t for t in input_range]
    start_date, end_date = range_list[0], range_list[-1]
    start_date += min_shift
    end_date += max_shift
    return start_date, end_date


def _extended_range_tuple_from_extended_range(input_range, min_shift, max_shift):
    """
    """
    range_list = [t for t in input_range]
    start_date, end_date = range_list[0], range_list[-1]
    return start_date, end_date


_EXTENDED_RANGE_TUPLE_RESOLUTION = {
    "base": _extended_range_tuple_from_base_range,
    "extended": _extended_range_tuple_from_extended_range,
}


_SERIES_CONSTRUCTOR_FACTORY = {
    "start_date": _series.Series.from_start_date_and_values,
    "range": _series.Series.from_dates_and_values,
}


_ARRAY_TRANSPOSER_FACTORY = {
    "vertical": _np.transpose,
    "horizontal": lambda x: x,
}


class Databox(
    _imports.DataboxImportMixin,
    _exports.DataboxExportMixin,
    _descriptions.DescriptionMixin,
    _views.ViewMixin,
    dict,
):
    """
    Unstructured data storage class
    """

    #[
    def __init__(
        self,
        /,
        description: str = "",
    ) -> None:
        self.set_description(description)

    @classmethod
    def from_dict(
        klass,
        _dict: dict,
        /,
    ) -> Self:
        """
        """
        self = klass()
        for k, v in _dict.items():
            self[k] = v
        return self

    @classmethod
    def from_array(
        klass,
        array: _np.ndarray,
        qid_to_name: Sequence[str] | dict[int, str],
        dates: _dates.Dater,
        /,
        target_databox: Self | None = None,
        qid_to_description: dict[int, str] | None = None,
        array_orientation: Literal["vertical", "horizontal", ] = "vertical",
        interpret_dates: Literal["start_date", "range", ] = "start_date",
    ) -> Self:
        """
        """
        self = target_databox or klass()
        constructor = _SERIES_CONSTRUCTOR_FACTORY[interpret_dates]
        transposer = _ARRAY_TRANSPOSER_FACTORY[array_orientation]
        for qid, data in enumerate(transposer(array)):
            name = qid_to_name.get(qid, None)
            if not name:
                continue
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            description = qid_to_description[qid] if qid_to_description else ""
            series = constructor(dates, data, description=description)
            self[name] = series
        return self

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
    ) -> Self:
        """
        """
        new_databox = _co.deepcopy(self)
        if source_names is None and target_names is None:
            return new_databox
        context_names = new_databox.get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        new_databox.rename(source_names, target_names)
        new_databox.keep(target_names)
        return new_databox

    def rename(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> None:
        """
        """
        context_names = self.get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        for old_name, new_name in (
            z for z in zip(source_names, target_names) if z[0] in context_names and z[0]!=z[1]
        ):
            self[new_name] = self.pop(old_name)

    def remove(
        self: Self,
        /,
        remove_names: SourceNames = None,
    ) -> None:
        """
        """
        if remove_names is None:
            return self
        context_names = self.get_names()
        remove_names, *_ = _resolve_source_target_names(remove_names, None, context_names)
        for n in set(remove_names).intersection(context_names):
            del self[n]

    def keep(
        self: Self,
        /,
        keep_names: SourceNames = None,
    ) -> None:
        """
        """
        if keep_names is None:
            return self
        context_names = self.get_names()
        keep_names, *_ = _resolve_source_target_names(keep_names, None, context_names)
        remove_names = set(context_names).difference(keep_names)
        self.remove(remove_names)

    def apply(
        self,
        func: Callable,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        in_place: bool = False,
        when_fails: Literal["error", "warning", "silent", ] = "error",
    ) -> None:
        """
        """
        context_names = self.get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        stream = _wrongdoings.STREAM_FACTORY[when_fails](f"Error(s) when applying function to Databox items:")
        for s, t in zip(source_names, target_names):
            try:
                self[t] = self[s]
                result = func(self[t])
                if not in_place:
                    self[t] = result
            except Exception as e:
                stream.add(f"{s} --> {t}: {repr(e)}", )
        stream._raise()

    def filter(
        self,
        name_test: Callable | None = None,
        value_test: Callable | None = None,
    ) -> Iterable[str]:
        """
        """
        names = self.get_names()
        if name_test is None and value_test is None:
            return names
        name_test = name_test if name_test else lambda x: True
        value_test = value_test if value_test else lambda x: True
        return [ n for n in names if name_test(n) and value_test(self[n]) ]

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

    def get_range_by_frequency(
        self,
        frequency: _dates.Frequency,
    ) -> Ranger:
        """
        Get the encompassing date range for all time series with the given frequency
        """
        if frequency == _dates.Frequency.UNKNOWN:
            return _dates.EmptyRanger()
        names = self.get_series_names_by_frequency(frequency)
        if not names:
            return _dates.EmptyRanger()
        min_start_date = min((self[n].start_date for n in names), key=_op.attrgetter("serial"))
        max_end_date = max((self[n].end_date for n in names), key=_op.attrgetter("serial"))
        return _dates.Ranger(min_start_date, max_end_date)

    def to_json(self, **kwargs):
        return _js.dumps(self, **kwargs)

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
            self_names = self.filter(value_test=lambda x: isinstance(x, _series.Series))
            other_names = other.filter(value_test=lambda x: isinstance(x, _series.Series))
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
        if new_start_date is None and new_end_date is None:
            return
        frequency = new_start_date.frequency if new_start_date is not None else new_end_date.frequency
        names = self.filter(value_test=lambda x: isinstance(x, _series.Series) and x.frequency == frequency)
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

    @classmethod
    def steady(
        klass,
        steady_databoxable: SteadyDataboxableProtocol,
        input_range: Iterable[_dates.Dater],
        /,
        deviation: bool = False,
        interpret_range: InterpretRange = "base",
    ) -> Self:
        """
        """
        self = klass()
        min_shift, max_shift = steady_databoxable.get_min_max_shifts()
        start_date, end_date = _resolve_input_range(input_range, min_shift, max_shift, interpret_range)
        items = steady_databoxable.generate_steady_items(start_date, end_date, deviation=deviation)
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
        kind = _quantities.QuantityKind.VARIABLE | _quantities.QuantityKind.SHOCK
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
    #]


Databank = Databox


def _resolve_source_target_names(
    source_names: SourceNames,
    target_names: TargetNames,
    context_names: Iterable[str],
    /,
) -> tuple[Iterable[str], Iterable[str]]:
    """
    """
    if source_names is None:
        source_names = context_names
    if isinstance(source_names, str):
        source_names = [ source_names ]
    if callable(source_names):
        func = source_names
        source_names = [ n for n in context_names if func(n) ]
    if target_names is None:
        target_names = source_names
    if isinstance(target_names, str):
        target_names = [ target_names ]
    if callable(target_names):
        func = target_names
        target_names = [ func(n) for n in source_names ]
    return source_names, target_names


def _resolve_input_range(
    input_range: Iterable[_dates.Dater],
    min_shift: int,
    max_shift: int,
    interpret_range: InterpretRange,
) -> tuple[Dater, Dater]:
    return _EXTENDED_RANGE_TUPLE_RESOLUTION[interpret_range](input_range, min_shift, max_shift)


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
