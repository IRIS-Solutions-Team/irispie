"""
Data management tools for storing and manipulating unstructured data
"""


#[

from __future__ import annotations

from typing import (Self, TypeAlias, Literal, Sequence, Protocol, Any, NoReturn, )
from collections.abc import (Iterable, Iterator, Callable, )
from numbers import (Number, )
import json as _js
import copy as _co
import types as _ty
import numpy as _np
import re as _re
import operator as _op
import functools as _ft
import itertools as _it
import os as _os
import documark as _dm

from ..conveniences import views as _views
from ..conveniences import descriptions as _descriptions
from ..conveniences import iterators as _iterators
from ..series.main import (Series, )
from ..dates import (Period, Frequency, Span, EmptySpan, )
from .. import quantities as _quantities
from .. import wrongdoings as _wrongdoings

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

    max_lag: int
    max_lead: int
    def generate_steady_items(self, *args) -> Any: ...

    #]


def _extended_span_tuple_from_base_span(
    input_span: Iterable[Period],
    min_shift: int,
    max_shift: int,
    prepend_initial: bool,
    append_terminal: bool,
    /,
) -> tuple[Period, Period]:
    """
    """
    range_list = tuple(t for t in input_span)
    start_date, end_date = range_list[0], range_list[-1]
    start_date += min_shift if prepend_initial else 0
    end_date += max_shift if append_terminal else 0
    return start_date, end_date


@_dm.reference(
    path=("data_management", "databoxes.md", ),
    categories={
        "constructor": "Creating new databoxes",
        "information": "Getting information about databoxes",
        "manipulation": "Manipulating databoxes",
        "import_export": "Importing and exporting databoxes",
    },
)
class Databox(
    _imports.Inlay,
    _exports.Inlay,
    _merge.Inlay,
    _views.Inlay,
    _dotters.DotterMixin,
    _fred.FredMixin,
    _descriptions.DescriptionMixin,
    dict,
):
    r"""
................................................................................

Databoxes
==========

`Databoxes` extend the standard `dict` class (technically, they are a
subclass), serving as a universal tool for storing and manipulating
unstructured data organized as key-value pairs. The values stored within
`Databoxes` can be of any type.

`Databoxes` can use any methods implemented for the standard `dict`
objects, and have additional functionalities for data item manipulation,
batch processing, importing and exporting data, and more.

................................................................................
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
        self._dotters = []
        self.__description__ = ""

    @classmethod
    @_dm.reference(
        category="constructor",
        call_name="Databox.empty",
    )
    def empty(
        klass,
        /,
    ) -> Self:
        """
················································································

==Create an empty Databox==

Generate a new, empty Databox instance. This class method is useful for 
initializing a Databox without any pre-existing data.

    Databox.empty()


### Input arguments ###

No input arguments are required for this method.


### Returns ###


???+ returns "Databox"
    Returns a new instance of an empty Databox.

················································································
        """
        return klass()

    @classmethod
    @_dm.reference(category="constructor", call_name="Databox.from_dict", )
    def from_dict(
        klass,
        _dict: dict,
        /,
    ) -> Self:
        """
················································································

==Create a new `Databox` from a `dict`==

Create a new Databox instance populated with data from a provided dictionary. 
This class method can be used to convert a standard Python dictionary into a 
Databox, incorporating all its functionalities.

    self = Databox.from_dict(_dict)


### Input arguments ###


???+ input "_dict"
    A dictionary containing the data to populate the new Databox. Each
    key-value pair in the dictionary will be an item in the Databox.


### Returns ###


???+ returns "self"
    Returns a new Databox populated with the contents of the provided
    dictionary.

················································································
        """
        self = klass()
        for k, v in _dict.items():
            self[k] = v
        return self

    @classmethod
    @_dm.reference(category="constructor", call_name="Databox.from_array", )
    def from_array(
        klass,
        array: _np.ndarray,
        names: Sequence[str],
        *,
        descriptions: Iterable[str] | None = None,
        periods: Iterable[Period] | None = None,
        start: Period | None = None,
        target_db: Self | None = None,
        orientation: Literal["vertical", "horizontal", ] = "vertical",
    ) -> Self:
        """
················································································

==Create a new `Databox` from a numpy array==

Convert a two-dimensional [numpy](https://numpy.org) array data into a
Databox, with the individual time series created from the rows or columns
of the numeric array.

    self = Databox.from_array(
        array,
        names,
        *,
        descriptions=None,
        periods=None,
        start=None,
        target_db=None,
        orientation="vertical",
    )


### Input arguments ###


???+ input "array"
    A numpy array containing the data to be included in the Databox.

???+ input "names"
    A sequence of names corresponding to the series in the array.

???+ input "descriptions"
    Descriptions for each series in the array.

???+ input "periods"
    An iterable of time periods corresponding to the rows of the array. Used if the data
    represents time series.

???+ input "start"
    The start period for the time series data. Used if 'periods' is not provided.

???+ input "target_db"
    An existing Databox to which the array data will be added. If `None`, a new 
    Databox is created.

???+ input "orientation"
    The orientation of the array, indicating how time series are arranged: 

    * `"horizontal"` means each row is a time series;

    * `"vertical"` means each column is a time series.


### Returns ###


???+ returns "self"
    Returns a new Databox populated with the data from the numpy array.

················································································
        """
        array = array if orientation == "horizontal" else array.T
        series_constructor = _get_series_constructor(start, periods, )
        return klass._from_horizontal_array_and_constructor(
            array,
            names,
            series_constructor,
            descriptions=descriptions,
            target_db=target_db,
        )

    @classmethod
    def _from_horizontal_array_and_constructor(
        klass,
        array: _np.ndarray,
        names: Iterable[str],
        series_constructor: Callable,
        *,
        descriptions: Sequence[str] | None = None,
        target_db: Self | None = None,
    ) -> Self:
        """
        """
        self = target_db or klass()
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

    @_dm.reference(category="information", )
    def get_names(self, /, ) -> list[str]:
        """
················································································

==Get all item names from a Databox==


    names = self.get_names()


### Input arguments ###


No input arguments are required for this method.


### Returns ###


???+ returns "names"
    A tuple containing all the names of items in the Databox.

················································································
        """
        return tuple(self.keys())

    @_dm.reference(category="information", )
    def get_missing_names(self, names: Iterable[str], ) -> tuple[str]:
        """
················································································

==Identify names not present in a Databox==

Find and return the names from a provided list that are not present in the 
Databox. This method is helpful for checking which items are missing or have 
yet to be added to the Databox.

    missing_names = self.get_missing_names(names)


### Input arguments ###


???+ input "names"
    An iterable of names to check against the Databox's items.


### Returns ###


???+ returns "missing_names"
    A tuple of names that are not found in the Databox.

················································································
        """
        return tuple(name for name in names if name not in self)

    @property
    @_dm.reference(category="property", )
    def num_items(self, /, ) -> int:
        """==Number of items in the databox=="""
        return len(self.keys())

    def to_dict(self: Self) -> dict:
        """
        Convert Databox to dict
        """
        return { k: v for k, v in self.items() }

    @_dm.reference(category="manipulation", )
    def copy(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        strict_names: bool = False,
    ) -> Self:
        """
................................................................................

==Create a copy of the Databox==

Produce a deep copy of the Databox, with options to filter and rename items 
during the duplication process.

    new_databox = self.copy(
        source_names=None,
        target_names=None,
        strict_names=False,
    )


### Input arguments ###


???+ input "source_names"
    Names of the items to include in the copy. Can be a list of names, a single 
    name, a callable returning `True` for names to include, or `None` to copy 
    all items.

???+ input "target_names"
    New names for the copied items, corresponding to 'source_names'. Can be a 
    list of names, a single name, or a callable function taking a source name 
    and returning the new target name.

???+ input "strict_names"
    If set to `True’, strictly adheres to the provided names, raising an error 
    if any source name is not found in the Databox.


### Returns ###


???+ returns "new_databox"
    A new Databox instance that is a deep copy of the current one, containing 
    either all items or only those specified.

................................................................................
        """
        new_databox = _co.deepcopy(self, )
        if source_names is None and target_names is None:
            return new_databox
        source_names, target_names, *_ = self._resolve_source_target_names(
            source_names, target_names, strict_names,
        )
        new_databox.rename(source_names, target_names, strict_names=strict_names, )
        new_databox.keep(target_names, strict_names=strict_names, )
        return new_databox

    def has(
        self: Self,
        /,
        names: Iterable[str] | str,
    ) -> bool:
        """
        """
        return (
            names in self if isinstance(names, str)
            else not self.get_missing_names(names, )
        )

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

    def print_contents(
        self,
        source_names: SourceNames = None,
    ) -> None:
        """
        """
        shallow = self.shallow(source_names=source_names, )
        content_view = shallow._get_content_view()
        print()
        print("\n".join(content_view, ))
        print()

    @_dm.reference(category="validation", )
    def validate(
        self: Self,
        validators: dict[str, Callable] | None,
        /,
        strict_names: bool = False,
        when_fails: Literal["critical", "error", "warning", "silent", ] = "error",
        title: str = "Databox validation errors",
        message_when_fails: str = "Failed validation",
        message_when_missing: str = "Missing item",
    ) -> None | NoReturn:
        r"""
        """
        if not validators:
            return
        when_fails_stream = _wrongdoings.create_stream(when_fails, title, )
        keys_to_validate = set(validators.keys())
        if not strict_names:
            keys_to_validate &= set(self.keys())
        for key in keys_to_validate:
            if key not in self:
                message = f"{message_when_missing}: {key}"
                when_fails_stream.add(message, )
                continue
            func = validators[key][0]
            result = func(self[key], ) if callable(func) else True
            if not result:
                message = validators[key][1] if len(validators[key]) > 1 else message_when_fails
                when_fails_stream.add(f"{message}: {key}", )
        when_fails_stream._raise()

    @_dm.reference(category="manipulation", )
    def rename(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
        strict_names: bool = False,
    ) -> None:
        """
················································································

==Rename items in a Databox==

Rename existing items in a Databox by specifying `source_names` and 
`target_names`. The `source_names` can be a list of names, a single name, or a 
callable function returning `True` for names to be renamed. Define `target_names`
as the new names for these items, either as a corresponding list, a single name,
or a callable function taking a source name and returning the new target name.

    self.rename(
        source_names=None,
        target_names=None,
        strict_names=False,
    )


### Input arguments ###


???+ input "source_names"
    The current names of the items to be renamed. Accepts a list of names, a 
    single name, or a callable that generates new names based on the given ones.

???+ input "target_names"
    The new names for the items. Should align with 'source_names'. Can be a list 
    of names, a single name, or a callable function taking each source name and 
    returning the corresponding target name.

???+ input "strict_names"
    If set to `True`, enforces strict adherence to the provided names, with an 
    error raised for any source name not found in the Databox.


### Returns ###

Returns `None`; `self` is modified in place.


················································································
        """
        source_names, target_names, *_ = self._resolve_source_target_names(
            source_names, target_names, strict_names,
        )
        for s, t in zip(source_names, target_names, ):
            self[t] = self.pop(s)

    @_dm.reference(category="manipulation", )
    def remove(
        self: Self,
        remove_names: SourceNames = None,
        *,
        strict_names: bool = False,
    ) -> None:
        """
················································································

==Remove specified items from a Databox==

Remove specified items from the Databox based on the provided names or a 
filtering function. Items to be removed can be specified as a list of names, a 
single name, a callable that returns `True` for names to be removed, or `None`.

    self.remove(
        remove_names=None,
        *,
        strict_names=False,
    )


### Input arguments ###


???+ input "remove_names"
    Names of the items to be removed from the Databox. Can be a list of names, a 
    single name, a callable that returns `True` for names to be removed, or 
    `None`. If `None`, no items are removed.

???+ input "strict_names"
    If `True`, strictly adheres to the provided names, raising an error if any 
    name is not found in the Databox.


### Returns ###

Returns `None`; `self` is modified in place.


················································································
        """
        if remove_names is None:
            return
        context_names = self.get_names()
        remove_names, *_ = self._resolve_source_target_names(
            remove_names, None, strict_names,
        )
        for n in remove_names:
            del self[n]


    @_dm.reference(category="manipulation", )
    def keep(
        self: Self,
        /,
        keep_names: SourceNames = None,
        strict_names: bool = False,
    ) -> None:
        r"""
················································································

==Keep specified items in a Databox==

Retain selected items in a Databox, removing all others. Specify the items to 
keep using `keep_names`, which can be a list of names, a single name, or a 
callable function determining which items to retain.

    self.keep(
        keep_names=None,
        strict_names=False,
    )


### Input arguments ###


???+ input "keep_names"
    The names of the items to be retained in the Databox. Can be a list of names, 
    a single name, or a callable function determining the items to keep.

???+ input "strict_names"
    If set to `True`, enforces strict adherence to the provided names, with an 
    error raised for any name not found in the Databox.


### Returns ###


???+ returns "None"
    Modifies the Databox in-place, keeping only the specified items, and does not 
    return a value.

················································································
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

    @_dm.reference(category="manipulation", )
    def apply(
        self,
        func: Callable,
        /,
        source_names: SourceNames = None,
        in_place: bool = True,
        when_fails: Literal["critical", "error", "warning", "silent", ] = "critical",
        strict_names: bool = False,
    ) -> None:
        r"""
················································································


==Apply a function to items in a Databox==

Apply a function to selected Databox items, either in place or by reassigning 
the results.

    self.apply(
        func,
        source_names=None,
        in_place=True,
        when_fails="critical",
        strict_names=False,
    )


### Input arguments ###


???+ input "func"
    The function to apply to each selected item in the Databox.

???+ input "source_names"
    Names of the items to which the function will be applied. Can be a list of 
    names, a single name, a callable returning `True’ for names to include, or 
    `None` to apply to all items.

???+ input "in_place"
    Determines if the results of the function should be assigned back to the 
    items in-place. If `True`, items are updated in-place; if `False`, the
    results are reassigned to the items.

???+ input "when_fails"
    Specifies the action to take if applying the function fails. Options are 
    "critical", "error", "warning", or "silent".

???+ input "strict_names"
    If set to `True`, strictly adheres to the provided names, raising an error 
    if any source name is not found in the Databox.


### Returns ###


???+ returns "None"
    Modifies items in the Databox in-place (note that the `in_place` input
    argument only applies to the Databox items, and not the Databox itself)
    and does not return a value. Errors are handled based on the
    `when_fails’ setting.


················································································
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

    def max_abs(
        self,
        other: Self,
    ) -> Self:
        """
        """
        def _max_abs(x, ):
            return _np.nanmax(abs(_np.array(x)))
        output = type(self)()
        for n in self.keys():
            if n not in other:
                continue
            numeric_classes = (Number, _np.ndarray, )
            if isinstance(self[n], Series, ) and isinstance(other[n], Series, ):
                diff_array = (self[n] - other[n]).get_data()
                output[n] = _max_abs(diff_array) if diff_array.size else None
            elif isinstance(self[n], numeric_classes, ) and isinstance(other[n], numeric_classes, ):
                output[n] = _max_abs(self[n] - other[n], )
        return output

    @_dm.reference(category="information", )
    def filter(
        self,
        /,
        name_test: Callable | None = None,
        value_test: Callable | None = None,
    ) -> Iterable[str]:
        r"""
················································································


==Filter items in a Databox==

Select Databox items based on custom name or value test functions.

    filtered_names = self.filter(
        name_test=None,
        value_test=None,
    )


### Input arguments ###


???+ input "name_test"
    A callable function to test each item's name. Returns `True` for names that 
    meet the specified condition.

???+ input "value_test"
    A callable function to test each item's value. Returns `True` for values that 
    meet the specified condition.


### Returns ###


???+ returns "filtered_names"
    A tuple of item names that meet the specified conditions.


················································································
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

    @_dm.reference(category="information", )
    def get_series_names_by_frequency(
        self,
        frequency: Frequency,
    ) -> tuple[str]:
        r"""
················································································

==Retrieve time series names by frequency==

Obtain a list of time series names that match a specified frequency.

    time_series_names = self.get_series_names_by_frequency(frequency)


### Input arguments ###


???+ input "self"
    The Databox object from which to retrieve time series names.

???+ input "frequency"
    The frequency to filter the time series names by. It should be a valid 
    frequency from the `irispie.Frequency` enumeration.


### Returns ###


???+ returns "time_series_names"
    A list of time series names in the Databox that match the specified frequency.

················································································
        """
        def _is_series_with_frequency(x, ):
            return isinstance(x, Series) and x.frequency == frequency
        return self.filter(value_test=_is_series_with_frequency, )

    @_dm.reference(category="information", )
    def get_span_by_frequency(
        self,
        frequency: Frequency,
    ) -> Span:
        r"""
················································································

==Retrieve the date span for time series by frequency==

Get the encompassing date span for all time series with a specified frequency.

    date_span = self.get_span_by_frequency(frequency)


### Input arguments ###


???+ input "self"
    The Databox object from which to retrieve the date span.

???+ input "frequency"
    The frequency for which to determine the date span. Can be an instance of 
    `irispie.Frequency` or a plain integer representing the frequency.


### Returns ###


???+ returns "date_span"
    The date span, as a `Span` object, encompassing all time series in the 
    Databox that match the specified frequency.

················································································
        """
        if frequency == Frequency.UNKNOWN:
            return EmptySpan()
        names = self.get_series_names_by_frequency(frequency)
        if not names:
            return EmptySpan()
        start_periods = (self[n].start_date for n in names)
        end_periods = (self[n].end_date for n in names)
        min_start_date = min(start_periods, key=_op.attrgetter("serial"), )
        max_end_date = max(end_periods, key=_op.attrgetter("serial"), )
        return Span(min_start_date, max_end_date, )

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
        self._lay(other, Series.overlay, **kwargs)

    def underlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> None:
        """
        """
        self._lay(other, Series.underlay, **kwargs)

    def _lay(
        self,
        other: Self,
        func: Callable,
        /,
        names: Iterable[str] | None = None,
        strict_names: bool = False,
        **kwargs,
    ) -> None:
        """"
        """
        if names is None:
            def value_test(x): return isinstance(x, Series)
            self_names = self.filter(value_test=value_test, )
            other_names = other.filter(value_test=value_test, )
            names = tuple(set(self_names) & set(other_names))
        if not strict_names:
            names = tuple(set(names) & set(self.keys()) & set(other.keys()))
        for n in names:
            if self[n].frequency == Frequency.UNKNOWN:
                continue
            if self[n].frequency != other[n].frequency:
                continue
            func(self[n], other[n], **kwargs, )

    def clip(
        self,
        /,
        new_start_date: Period | None = None,
        new_end_date: Period | None = None,
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
        value_test = lambda x: isinstance(x, Series) and x.frequency == frequency
        names = self.filter(value_test=value_test, )
        for n in names:
            self[n].clip(new_start_date, new_end_date, )

    def prepend(
        self,
        prepending: Self,
        end_prepending: Period,
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
        context = (dict(context) if context else {}) | { k: v for k, v in self.items() }
        return eval(expression, context, )

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
        span: Iterable[Period],
        *,
        deviation: bool = False,
        prepend_initial: bool = True,
        append_terminal: bool = True,
    ) -> Self:
        """
        """
        self = klass()
        start, end = _extended_span_tuple_from_base_span(
            span,
            steady_databoxable.max_lag,
            steady_databoxable.max_lead,
            prepend_initial,
            append_terminal,
        )
        items = steady_databoxable.generate_steady_items(
            start, end,
            deviation=deviation,
        )
        return klass({ k: v for k, v in items })

    zero = _ft.partialmethod(steady, deviation=True, )

    def minus_control(
        self,
        model,
        control: Self,
    ) -> None:
        """
        """
        name_to_minus_control_func = model.map_name_to_minus_control_func()
        for name, func in name_to_minus_control_func.items():
            description = self[name].get_description()
            self[name] = func(self[name], control[name], )
            self[name].set_description(description, )

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
            source_names = (source_names, )
        if callable(source_names):
            func = source_names
            source_names = tuple(n for n in context_names if func(n))
        if target_names is None:
            target_names = source_names
        if isinstance(target_names, str):
            target_names = (target_names, )
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
    start: Period | None = None,
    periods: Iterable[Period] | None = None,
    /,
) -> Callable | None:
    """
    """
    #[
    if start is not None:
        return lambda values: Series(start=start, values=values, )
    elif periods is not None:
        return lambda values: Series(periods=periods, values=values, )
    #]

