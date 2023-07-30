"""
"""


#[
from __future__ import annotations
# from IPython import embed

import json as js_
import copy as co_
import types as ty_
import numpy as np_
import re as re_
import operator as op_
import functools as ft_
from typing import (Self, TypeAlias, Literal, Sequence, Protocol, Any, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )

from ..dataman import (dates as da_, )
from ..dataman import (series as se_, )
from ..dataman import (views as vi_, )
from ..dataman import (imports as im_, )
from ..dataman import (exports as ex_, )
from .. import (quantities as qu_, )
from ..mixins import (userdata as ud_, )
#]


__all__ = [
    "Databank",
]


SourceNames: TypeAlias = Iterable[str] | str | Callable[[str], bool] | None
TargetNames: TypeAlias = Iterable[str] | str | Callable[[str], str] | None
InterpretRange: TypeAlias = Literal["base"] | Literal["extended"]


class SteadyDatabankableProtocol(Protocol):
    """
    """
    #[
    def _get_min_max_shifts(self, *args) -> Any: ...
    def _get_steady_databank(self, *args) -> Any: ...
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


_SERIES_CONSTRUCTOR_RESOLUTION = {
    "start_date": se_.Series.from_start_date_and_data,
    "range": se_.Series.from_dates_and_data,
}


_ARRAY_TRANSPOSER_RESOLUTION = {
    "vertical": np_.transpose,
    "horizontal": lambda x: x,
}


class Databank(
    im_.DatabankImportMixin,
    ex_.DatabankExportMixin,
    ud_.DescriptionMixin,
    vi_.DatabankViewMixin,
    ty_.SimpleNamespace,
):
    """
    Create a databank object as a simple namespace with utility functions
    """
    #[
    def __init__(
        self,
        /,
        description: str = "",
    ) -> None:
        self.set_description(description)

    @classmethod
    def _from_dict(
        cls,
        _dict: dict,
        /,
    ) -> Self:
        """
        """
        self = cls()
        for k, v in _dict.items():
            self.__setattr__(k, v)
        return self

    @classmethod
    def _from_array(
        cls,
        array: np_.ndarray,
        qid_to_name: Sequence[str] | dict[int, str],
        dates: da_.Dater,
        /,
        add_to_databank: Self | None = None,
        qid_to_description: dict[int, str] | None = None,
        array_orientation: Literal["vertical"] | Literal["horizontal"] = "vertical",
        interpret_dates: Literal["start_date"] | Literal["range"] = "start_date",
    ) -> Self:
        """
        """
        self = add_to_databank if add_to_databank else cls()
        constructor = _SERIES_CONSTRUCTOR_RESOLUTION[interpret_dates]
        transposer = _ARRAY_TRANSPOSER_RESOLUTION[array_orientation]
        for qid, data in enumerate(transposer(array)):
            name = qid_to_name.get(qid, None)
            if not name:
                continue
            description = qid_to_description[qid] if qid_to_description else ""
            series = constructor(dates, data.reshape(-1, 1), description=description)
            setattr(self, name, series)
        return self

    def _name_test_(self, n) -> bool:
        return not n.startswith("_") and not isinstance(getattr(self, n), type(self.__init__))

    def _get_names(self: Self) -> Iterable[str]:
        """
        Get all names stored in a databank save for private attributes
        """
        return [ n for n in dir(self) if self._name_test_(n) ]

    def _get_num_records(self: Self) -> int:
        """
        """
        return sum(1 for n in dir(self) if self._name_test_(n) )

    def _to_dict(self: Self) -> dict:
        """
        Convert Databank to dict
        """
        return vars(self)

    def _copy(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> Self:
        """
        """
        new_databank = co_.deepcopy(self)
        new_databank = new_databank._rename(source_names, target_names)
        new_databank._keep(target_names)
        return new_databank

    def _rename(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> Self:
        """
        """
        context_names = self._get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        for old_name, new_name in (
            z for z in zip(source_names, target_names) if z[0] in context_names and z[0]!=z[1]
        ):
            self.__dict__[new_name] = self.__dict__.pop(old_name)
        return self

    def _remove(
        self: Self,
        /,
        remove_names: SourceNames = None,
    ) -> Self:
        """
        """
        if remove_names is None:
            return self
        context_names = self._get_names()
        remove_names, *_ = _resolve_source_target_names(remove_names, None, context_names)
        for n in set(remove_names).intersection(context_names):
            del self.__dict__[n]
        return self

    def _keep(
        self: Self,
        /,
        keep_names: SourceNames = None,
    ) -> Self:
        """
        """
        if keep_names is None:
            return self
        context_names = self._get_names()
        keep_names, *_ = _resolve_source_target_names(keep_names, None, context_names)
        remove_names = set(context_names).difference(keep_names)
        return self._remove(remove_names)

    def _apply(
        self,
        func: Callable,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> Self:
        context_names = self._get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        for s, t in zip(source_names, target_names):
            setattr(self, t, func(getattr(self, s)))
        return self


    def _filter(
        self,
        name_test: Callable | None = None,
        value_test: Callable | None = None,
    ) -> Iterable[str]:
        """
        """
        names = self._get_names()
        if name_test is None and value_test is None:
            return names
        name_test = name_test if name_test else lambda x: True
        value_test = value_test if value_test else lambda x: True
        return [ n for n in names if name_test(n) and value_test(getattr(self, n)) ]

    def _get_series_names_by_frequency(
        self,
        frequency: da_.Frequency,
    ) -> Iterable[str]:
        """
        """
        return self._filter(value_test=lambda x: isinstance(x, se_.Series) and x.frequency==frequency)

    def _get_range_by_frequency(
        self,
        frequency: da_.Frequency,
    ) -> Ranger:
        names = self._get_series_names_by_frequency(frequency)
        if not names:
            return Ranger(None, None)
        min_start_date = min((getattr(self, n).start_date for n in names), key=op_.attrgetter("serial"))
        max_end_date = max((getattr(self, n).end_date for n in names), key=op_.attrgetter("serial"))
        return da_.Ranger(min_start_date, max_end_date)

    def _to_json(self, **kwargs):
        return js_.dumps(vars(self), **kwargs)

    def _underlay(self, other) -> None:
        """"
        """
        self_names = self._filter(value_test=lambda x: isinstance(x, se_.Series))
        other_names = other._filter(value_test=lambda x: isinstance(x, se_.Series))
        names = set(self_names).intersection(other_names)
        for n in names:
            self_n = getattr(self, n)
            other_n = getattr(other, n)
            if self_n.frequency == other_n.frequency:
                self_n.underlay(other_n)

    def _clip(
        self,
        new_start_date: da_.Dater | None = None,
        new_end_date: da_.Dater | None = None,
    ) -> None:
        if new_start_date is None and new_end_date is None:
            return
        frequency = new_start_date.frequency if new_start_date is not None else new_end_date.frequency
        names = self._filter(value_test=lambda x: isinstance(x, se_.Series) and x.frequency == frequency)
        for n in names:
            x = getattr(self, n)
            x.clip(new_start_date, new_end_date)

    def _add_steady(
        self,
        steady_databankable: SteadyDatabankableProtocol,
        input_range: Iterable[da_.Dater],
        /,
        deviation: bool = False,
        interpret_range: InterpretRange = "base",
    ) -> Self:
        """
        """
        min_shift, max_shift = steady_databankable._get_min_max_shifts()
        start_date, end_date = _resolve_input_range(input_range, min_shift, max_shift, interpret_range)
        num_columns = int(end_date - start_date + 1)
        if num_columns < 1:
            raise Exception("Empty date range is not allowed when creating steady databank")
        steady_databank = steady_databankable._get_steady_databank(start_date, end_date, deviation=deviation)
        self._update(steady_databank)

    _add_zero = ft_.partialmethod(_add_steady, deviation=True)

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value) -> None:
        self.__dict__[name] = value

    def __or__(self, other) -> Self:
        new = co_.deepcopy(self)
        new.__dict__.update(other.__dict__)
        return new

    def _update(
        self,
        other: Databank,
        /,
    ) -> Self:
        """
        Update self using records from other
        """
        self.__dict__.update(other.__dict__)
    #]


#
# Add databank methods without the leading underscore
#
single_underscore_names = [
    n for n in dir(Databank) 
    if n.startswith("_") and not n.startswith("__") and not n.endswith("_")
]
for n in single_underscore_names:
    setattr(Databank, n[1:], getattr(Databank, n))


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
    input_range: Iterable[da_.Dater],
    min_shift: int,
    max_shift: int,
    interpret_range: InterpretRange,
) -> tuple[Dater, Dater]:
    return _EXTENDED_RANGE_TUPLE_RESOLUTION[interpret_range](input_range, min_shift, max_shift)

