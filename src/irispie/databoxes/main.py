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
    A protocol defining the interface for objects that can generate steady 
    databox items.
    """
    #[

    max_lag: int
    max_lead: int
    def generate_steady_items(self, *args) -> Any: ...
        """
        Generate items with steady lag and lead properties.
        """
    #]


def _extended_span_tuple_from_base_span(
    input_span: Iterable[Period],
    min_shift: int,
    max_shift: int,
    prepend_initial: bool,
    append_terminal: bool,
    /,
) -> tuple[Period, Period]:
    r"""
    ................................................................................
    ==Calculate Extended Span from Base Span==

    Computes an extended span tuple based on an input span of periods, applying 
    optional shifts to the start and end periods. This function is useful for 
    defining expanded date ranges for time series data.

    The span is adjusted using `min_shift` and `max_shift`, which represent 
    offsets applied to the start and end of the input span, respectively. These 
    offsets are conditionally applied based on the `prepend_initial` and 
    `append_terminal` flags.

    ................................................................................

    ### Input arguments ###
    ???+ input "input_span"
        An iterable of `Period` objects representing the base span. The first and 
        last elements of this iterable define the initial start and end dates.

    ???+ input "min_shift"
        An integer specifying the number of periods to shift the start date. A 
        positive value moves the start date forward, and a negative value moves 
        it backward.

    ???+ input "max_shift"
        An integer specifying the number of periods to shift the end date. A 
        positive value moves the end date forward, and a negative value moves 
        it backward.

    ???+ input "prepend_initial"
        A boolean flag indicating whether to apply the `min_shift` to the start 
        date. If `False`, the start date remains unchanged.

    ???+ input "append_terminal"
        A boolean flag indicating whether to apply the `max_shift` to the end 
        date. If `False`, the end date remains unchanged.

    ### Returns ###
    ???+ returns "tuple[Period, Period]"
        A tuple containing two `Period` objects: the adjusted start and end dates.

    ### Example ###
    ```python
        input_span = [Period("2023-01"), Period("2023-12")]
        extended_span = _extended_span_tuple_from_base_span(
            input_span, min_shift=-2, max_shift=3, 
            prepend_initial=True, append_terminal=True
        )
        print(extended_span)  # Output: (Period("2022-11"), Period("2024-03"))
    ```
    ................................................................................
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
        r"""
        ................................................................................
        ==Constructor==

        Initializes a `Databox` object. This method sets up the internal attributes 
        and prepares the object for data storage and manipulation.

        ................................................................................

        ### Input arguments ###
        ???+ input "*args"
            Positional arguments passed to the base dictionary.

        ???+ input "**kwargs"
            Keyword arguments passed to the base dictionary.

        ### Returns ###
        ???+ returns "None"
            This constructor initializes the instance and does not return a value.

        ................................................................................
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
        r"""
        ................................................................................
        ==Create an Empty Databox==

        Generate a new, empty `Databox` instance.

        This class method provides a standardized way to initialize an empty 
        `Databox` for further configuration.

        ................................................................................

        ### Input arguments ###
        No input arguments are required for this method.

        ### Returns ###
        ???+ returns "Databox"
            A new, empty instance of `Databox`.

        ### Example ###
        ```python
            empty_box = Databox.empty()
            print(empty_box)
        ```
        ................................................................................
        """
        return klass()

    @classmethod
    @_dm.reference(category="constructor", call_name="Databox.from_dict", )
    def from_dict(
        klass,
        _dict: dict,
        /,
    ) -> Self:
        r"""
        ................................................................................
        ==Create a Databox from a Dictionary==

        Convert a standard Python dictionary into a `Databox`.

        This method initializes a `Databox` populated with data from the provided 
        dictionary, preserving all key-value pairs.

        ................................................................................

        ### Input arguments ###
        ???+ input "_dict"
            A dictionary containing data to populate the new `Databox`.

        ### Returns ###
        ???+ returns "Databox"
            A new `Databox` populated with the dictionary contents.

        ### Example ###
        ```python
            data_dict = {"a": 1, "b": 2}
            databox = Databox.from_dict(data_dict)
            print(databox)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Construct Databox from Array==

        An internal method to build a `Databox` from a horizontally structured 
        numpy array, using a series constructor to format the data.

        This method facilitates the `from_array` functionality by enabling 
        detailed construction logic.

        ................................................................................

        ### Input arguments ###
        ???+ input "array"
            A numpy array containing the data to populate the `Databox`.

        ???+ input "names"
            An iterable of strings specifying the names for the data series.

        ???+ input "series_constructor"
            A callable used to construct each data series.

        ???+ input "descriptions"
            Optional descriptions for each series.

        ???+ input "target_db"
            A target `Databox` instance to populate. If `None`, a new `Databox` is created.

        ### Returns ###
        ???+ returns "Databox"
            A `Databox` populated with the array data.

        ### Example ###
        ```python
            data = _np.array([[1, 2], [3, 4]])
            db = Databox._from_horizontal_array_and_constructor(
                data, ["series1", "series2"], lambda x: x
            )
            print(db)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Iterate Over Variants==

        Yields dictionaries containing variants of the `Databox` items. Variants are 
        generated by iterating over the items using the provided or default iterators.

        This method allows traversal of various representations of the `Databox` items, 
        making it suitable for scenarios requiring multiple transformations or 
        permutations of the data.

        ................................................................................

        ### Input arguments ###
        ???+ input "item_iterator"
            An optional iterator for processing individual items. If `None`, a 
            default item iterator is used.

        ???+ input "names"
            An optional iterable of item names to include in the iteration. If `None`, 
            all item names are considered.

        ### Returns ###
        ???+ returns "Iterator[dict]"
            An iterator that yields dictionaries containing item variants.

        ### Example ###
        ```python
            variants = databox.iter_variants(names=["a", "b"])
            for variant in variants:
                print(variant)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Retrieve Item Names==

        Returns a list of all the item names (keys) present in the `Databox`.

        This method provides a straightforward way to access the names of all stored 
        data items, aiding in operations like filtering and validation.

        ................................................................................

        ### Input arguments ###
        No input arguments are required for this method.

        ### Returns ###
        ???+ returns "list[str]"
            A list of all item names (keys) in the `Databox`.

        ### Example ###
        ```python
            names = databox.get_names()
            print(names)
        ```
        ................................................................................
        """
        return tuple(self.keys())

    @_dm.reference(category="information", )
    def get_missing_names(self, names: Iterable[str], ) -> tuple[str]:
        r"""
        ................................................................................
        ==Identify Missing Names==

        Determines which names from a given list are not present in the `Databox`.

        This method is useful for validating that required data items are available 
        in the `Databox`.

        ................................................................................

        ### Input arguments ###
        ???+ input "names"
            An iterable of strings representing the names to check against the 
            `Databox` keys.

        ### Returns ###
        ???+ returns "tuple[str]"
            A tuple containing the names that are missing from the `Databox`.

        ### Example ###
        ```python
            required_names = ["key1", "key2"]
            missing = databox.get_missing_names(required_names)
            print(missing)
        ```
        ................................................................................
        """
        return tuple(name for name in names if name not in self)

    @property
    @_dm.reference(category="property", )
    def num_items(self, /, ) -> int:
        r"""
        ................................................................................
        ==Number of Items==

        A property that returns the number of items (key-value pairs) in the `Databox`.

        This property is helpful for quickly determining the size of the `Databox`.

        ................................................................................

        ### Input arguments ###
        No input arguments are required for this property.

        ### Returns ###
        ???+ returns "int"
            The number of items in the `Databox`.

        ### Example ###
        ```python
            count = databox.num_items
            print(count)
        ```
        ................................................................................
        """
        return len(self.keys())

    def to_dict(self: Self) -> dict:
        r"""
        ................................................................................
        ==Convert to Dictionary==

        Converts the `Databox` into a standard Python dictionary.

        This method is useful for interoperability with libraries or tools that 
        require native dictionary structures.

        ................................................................................

        ### Input arguments ###
        No input arguments are required for this method.

        ### Returns ###
        ???+ returns "dict"
            A dictionary containing all the items from the `Databox`.

        ### Example ###
        ```python
            native_dict = databox.to_dict()
            print(native_dict)
        ```
        ................................................................................
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
       r"""
................................................................................

==Create a copy of the Databox==

Produce a deep copy of the Databox, with options to filter and rename items 
during the duplication process.
        This method is particularly useful when creating subsets or alternative 
        versions of the data without affecting the original.

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
        r"""
        ................................................................................
        ==Check for Item Existence==

        Determines whether the `Databox` contains one or more specified items.

        ................................................................................

        ### Input arguments ###
        ???+ input "names"
            A single name or an iterable of names to check for existence in the 
            `Databox`.

        ### Returns ###
        ???+ returns "bool"
            `True` if all specified names exist in the `Databox`, otherwise `False`.

        ### Example ###
        ```python
            exists = databox.has(["key1", "key2"])
            print(exists)  # True or False
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Create a Shallow Copy of the Databox==

        Generates a shallow copy of the `Databox`, optionally renaming items or 
        restricting the copy to specific items.

        A shallow copy retains references to the original data objects instead of 
        duplicating them.

        ................................................................................

        ### Input arguments ###
        ???+ input "source_names"
            An optional list, single name, or callable function specifying the items 
            to include in the shallow copy. Defaults to all items.

        ???+ input "target_names"
            An optional list, single name, or callable function specifying new names 
            for the copied items. Corresponds to `source_names`.

        ???+ input "strict_names"
            A boolean flag. If `True`, raises an error for any `source_names` not found 
            in the `Databox`.

        ### Returns ###
        ???+ returns "Databox"
            A new `Databox` instance containing the shallow copy of items.

        ### Example ###
        ```python
            shallow_copy = databox.shallow(source_names=["a", "b"])
            print(shallow_copy)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Print Databox Contents==

        Outputs the contents of the `Databox` to the console, formatted for readability.

        This method is useful for debugging and inspecting specific items in the 
        `Databox`.

        ................................................................................

        ### Input arguments ###
        ???+ input "source_names"
            An optional list, single name, or callable function specifying the items 
            to display. If `None`, all items are shown.

        ### Returns ###
        ???+ returns "None"
            Prints the contents to the console. No value is returned.

        ### Example ###
        ```python
            databox.print_contents(source_names=["key1", "key2"])
        ```
        ................................................................................
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
        ................................................................................
        ==Validate Items in the Databox==

        Validates items in the `Databox` against a set of provided validator functions. 
        Errors or warnings are logged based on the `when_fails` setting.

        This method ensures that items meet specific criteria, facilitating data 
        quality checks before processing.

        ................................................................................

        ### Input arguments ###
        ???+ input "validators"
            A dictionary where keys are item names and values are callables used to 
            validate the corresponding items.

        ???+ input "strict_names"
            A boolean flag. If `True`, enforces that all keys in `validators` must be 
            present in the `Databox`.

        ???+ input "when_fails"
            Specifies the action to take if validation fails:
            
            * `"critical"`: Raises a critical error.
            * `"error"`: Raises a standard error.
            * `"warning"`: Logs a warning.
            * `"silent"`: Suppresses all output.

        ???+ input "title"
            The title for the validation error log.

        ???+ input "message_when_fails"
            The message to display when validation fails for an item.

        ???+ input "message_when_missing"
            The message to display when an item is missing.

        ### Returns ###
        ???+ returns "None or NoReturn"
            Does not return a value. Behavior depends on `when_fails`.

        ### Example ###
        ```python
            validators = {
                "key1": lambda x: isinstance(x, int),
                "key2": lambda x: x > 0
            }
            databox.validate(validators, when_fails="warning")
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Compute Maximum Absolute Difference==

        Calculates the maximum absolute difference between corresponding items in 
        the current `Databox` and another `Databox`. The result is stored in a new 
        `Databox`.

        This method is designed to analyze discrepancies between two data collections, 
        such as detecting deviations in numerical series.

        ................................................................................

        ### Input arguments ###
        ???+ input "other"
            Another `Databox` instance to compare against the current instance. Must 
            have overlapping keys with comparable numerical data.

        ### Returns ###
        ???+ returns "Databox"
            A new `Databox` containing the maximum absolute differences for each key.

        ### Example ###
        ```python
            max_diff = databox1.max_abs(databox2)
            print(max_diff)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Export Databox to JSON==

        Serializes the `Databox` to a JSON file, storing its contents in a format 
        compatible with external tools and applications.

        ................................................................................

        ### Input arguments ###
        ???+ input "file_name"
            The file path where the JSON representation of the `Databox` will be saved.

        ???+ input "**kwargs"
            Additional keyword arguments passed to the `json.dump` function for 
            customization of the serialization process.

        ### Returns ###
        ???+ returns "None"
            This method writes the serialized JSON data to a file and does not return 
            a value.

        ### Example ###
        ```python
            databox.to_json("output.json", indent=4)
        ```
        ................................................................................
        """
        with open(file_name, "wt+") as f:
            return _js.dump(self, f, **kwargs)

    def overlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Overlay Another Databox==

        Combines the current `Databox` with another by applying an overlay operation. 
        This method updates items in the current `Databox` using corresponding items 
        from the `other` `Databox`.

        ................................................................................

        ### Input arguments ###
        ???+ input "other"
            Another `Databox` instance whose items will overlay the current `Databox`.

        ???+ input "**kwargs"
            Additional arguments passed to the overlay operation.

        ### Returns ###
        ???+ returns "None"
            The current `Databox` is modified in place.

        ### Example ###
        ```python
            databox1.overlay(databox2)
        ```
        ................................................................................
        """
        self._lay(other, Series.overlay, **kwargs)

    def underlay(
        self,
        other: Self,
        /,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Underlay Another Databox==

        Combines the current `Databox` with another by applying an underlay operation. 
        This method retains items in the current `Databox` and integrates 
        non-conflicting items from the `other` `Databox`.

        ................................................................................

        ### Input arguments ###
        ???+ input "other"
            Another `Databox` instance whose items will underlay the current `Databox`.

        ???+ input "**kwargs"
            Additional arguments passed to the underlay operation.

        ### Returns ###
        ???+ returns "None"
            The current `Databox` is modified in place.

        ### Example ###
        ```python
            databox1.underlay(databox2)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Generic Layering Operation==

        An internal method for layering two `Databox` instances using a specified 
        operation. This can be used for both overlay and underlay logic.

        ................................................................................

        ### Input arguments ###
        ???+ input "other"
            Another `Databox` instance to layer with the current instance.

        ???+ input "func"
            A callable function defining the layering operation to apply.

        ???+ input "names"
            An optional list of item names to restrict the layering operation. If 
            `None`, all items are considered.

        ???+ input "strict_names"
            A boolean flag. If `True`, raises an error if any `names` are not found 
            in both `Databox` instances.

        ???+ input "**kwargs"
            Additional arguments passed to the layering function.

        ### Returns ###
        ???+ returns "None"
            Modifies the current `Databox` in place.

        ### Example ###
        ```python
            databox1._lay(databox2, Series.overlay, names=["key1", "key2"])
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Clip Time Series to a Date Range==

        Trims all time series in the `Databox` to fit within the specified start and 
        end dates.

        ................................................................................

        ### Input arguments ###
        ???+ input "new_start_date"
            The new start date for the time series. If `None`, the current start date 
            is retained.

        ???+ input "new_end_date"
            The new end date for the time series. If `None`, the current end date is 
            retained.

        ### Returns ###
        ???+ returns "None"
            Modifies the time series in the `Databox` in place.

        ### Example ###
        ```python
            databox.clip(new_start_date=Period("2023-01"), new_end_date=Period("2023-12"))
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Prepend Data to a Databox==

        Prepends the contents of another `Databox` up to a specified end date. This 
        operation is useful for extending a `Databox` with historical data.

        ................................................................................

        ### Input arguments ###
        ???+ input "prepending"
            A `Databox` instance containing the data to prepend.

        ???+ input "end_prepending"
            The last date from the `prepending` `Databox` to include in the operation.

        ### Returns ###
        ???+ returns "Databox"
            The updated `Databox` with the prepended data.

        ### Example ###
        ```python
            updated_databox = databox.prepend(prepending_databox, Period("2022-12"))
            print(updated_databox)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Evaluate an Expression==

        Evaluates a string expression within the context of the `Databox` and an 
        optional user-provided context.

        This method supports accessing `Databox` items directly by name and performing 
        calculations or operations involving those items.

        ................................................................................

        ### Input arguments ###
        ???+ input "expression"
            A string representing the expression to evaluate.

        ???+ input "context"
            An optional dictionary providing additional variables or functions to use 
            during evaluation.

        ### Returns ###
        ???+ returns "Any"
            The result of evaluating the expression.

        ### Example ###
        ```python
            result = databox.evaluate_expression("key1 + key2")
            print(result)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Evaluate a Databox Expression==

        Parses and evaluates a Python expression within the `Databox`'s context. This 
        method provides a mechanism for dynamic calculations involving `Databox` items.

        ................................................................................

        ### Input arguments ###
        ???+ input "expression"
            A string containing the Python expression to evaluate.

        ???+ input "context"
            An optional dictionary providing additional variables or functions for use 
            during evaluation.

        ### Returns ###
        ???+ returns "Any"
            The result of the evaluated expression.

        ### Example ###
        ```python
            result = databox.eval("sum([key1, key2])")
            print(result)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Call Databox for Evaluation==

        Enables the `Databox` to be directly called with an expression for evaluation. 
        This method simplifies dynamic calculations by acting as a shorthand for the 
        `eval` method.

        ................................................................................

        ### Input arguments ###
        ???+ input "expression"
            A string containing the expression to evaluate.

        ???+ input "context"
            An optional dictionary providing additional variables or functions for use 
            during evaluation.

        ### Returns ###
        ???+ returns "Any"
            The result of the evaluated expression.

        ### Example ###
        ```python
            result = databox("key1 + key2")
            print(result)
        ```
        ................................................................................
        """
        return self.eval(expression, context, )

    @classmethod
    def steady(
        klass,
        steady_databoxable: SteadyDataboxableProtocol,
        span: Iterable[Period],
        *,
        deviation: bool = False,
        unpack_single: bool = True,
        prepend_initial: bool = True,
        append_terminal: bool = True,
    ) -> Self:
        r"""
        ................................................................................
        ==Create a Steady Databox==

        Constructs a `Databox` by generating items based on a steady protocol and a 
        specified time span. This method provides a way to create predictive or 
        interpolative datasets.

        ................................................................................

        ### Input arguments ###
        ???+ input "steady_databoxable"
            An object implementing the `SteadyDataboxableProtocol` for generating 
            steady items.

        ???+ input "span"
            An iterable of `Period` objects defining the time range.

        ???+ input "deviation"
            A boolean indicating whether to include deviation data in the generated 
            items. Default is `False`.

        ???+ input "unpack_single"
            A boolean specifying whether to unpack single-item results. Default is 
            `True`.

        ???+ input "prepend_initial"
            A boolean indicating whether to include initial padding in the time span. 
            Default is `True`.

        ???+ input "append_terminal"
            A boolean indicating whether to include terminal padding in the time span. 
            Default is `True`.

        ### Returns ###
        ???+ returns "Databox"
            A new `Databox` populated with steady items.

        ### Example ###
        ```python
            steady_box = Databox.steady(
                steady_databoxable, span, deviation=True
            )
            print(steady_box)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Subtract Control from Model Data==

        Adjusts the `Databox` items based on a control dataset by subtracting the 
        control values using a model-defined operation. This method is useful for 
        comparative analysis, such as isolating effects or deviations.

        ................................................................................

        ### Input arguments ###
        ???+ input "model"
            A model object that provides the mapping of names to subtraction functions.

        ???+ input "control"
            A `Databox` instance containing the control dataset to subtract.

        ### Returns ###
        ???+ returns "None"
            Modifies the `Databox` items in place.

        ### Example ###
        ```python
            databox.minus_control(model, control_databox)
        ```
        ................................................................................
        """
        name_to_minus_control_func = model.map_name_to_minus_control_func()
        for name, func in name_to_minus_control_func.items():
            description = self[name].get_description()
            self[name] = func(self[name], control[name], )
            self[name].set_description(description, )

    def __or__(self, other) -> Self:
        r"""
        ................................................................................
        ==Merge Two Databoxes==

        Implements the bitwise OR operator (`|`) to merge two `Databox` instances. 
        The resulting `Databox` contains all items from both instances, with items 
        from `other` overriding those in the current instance in case of conflicts.

        ................................................................................

        ### Input arguments ###
        ???+ input "other"
            Another `Databox` instance to merge with the current instance.

        ### Returns ###
        ???+ returns "Databox"
            A new `Databox` instance containing the merged items.

        ### Example ###
        ```python
            merged_box = databox1 | databox2
            print(merged_box)
        ```
        ................................................................................
        """
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
        r"""
        ................................................................................
        ==Resolve Source and Target Names==

        An internal utility method to resolve source and target names for operations 
        like renaming, filtering, or mapping items in the `Databox`. Handles multiple 
        input types such as lists, callables, or single names.

        ................................................................................

        ### Input arguments ###
        ???+ input "source_names"
            The source names to resolve. Can be a list, a single name, or a callable 
            function.

        ???+ input "target_names"
            The target names to resolve. Should correspond to `source_names` in 
            structure and order.

        ???+ input "strict_names"
            A boolean flag. If `True`, enforces strict resolution and raises an error 
            if any source name is not found in the `Databox`.

        ### Returns ###
        ???+ returns "tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]"
            A tuple containing resolved source names, target names, and the complete 
            set of item names in the `Databox`.

        ### Example ###
        ```python
            sources, targets, all_names = databox._resolve_source_target_names(
                source_names=["key1"], target_names=["keyA"]
            )
            print(sources, targets, all_names)
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Apply Function to Item==

    Applies a provided function to the `source` item and stores the result in the 
    `target`. This utility function is designed for item-level transformations 
    within a `Databox`.

    ................................................................................

    ### Input arguments ###
    ???+ input "func"
        A callable function to apply to the `source` item.

    ???+ input "source"
        The input item on which the function is applied.

    ???+ input "target"
        The output item where the result of the function is stored.

    ### Returns ###
    ???+ returns "None"
        Modifies the `target` in place with the result of applying the function.

    ### Example ###
    ```python
        _apply_to_item(lambda x: x * 2, 5, target_variable)
    ```
    ................................................................................
    """
    target = source
    return func(target)


def _reformat_eval_expression(expression: str, ) -> str:
    r"""
    ................................................................................
    ==Reformat Evaluation Expression==

    Rewrites an evaluation expression to handle specific cases, such as equality 
    operators, in the context of the `Databox`. This function ensures compatibility 
    with Python's `eval` function.

    ................................................................................

    ### Input arguments ###
    ???+ input "expression"
        A string representing the expression to reformat.

    ### Returns ###
    ???+ returns "str"
        The reformatted expression, ready for evaluation.

    ### Example ###
    ```python
        reformatted = _reformat_eval_expression("a = b")
        print(reformatted)  # Outputs: (a)-(b)
    ```
    ................................................................................
    """
    #[
    if "=" in expression:
        lhs, *rhs = expression.split(expression, "=", )
        rhs = "=".join(rhs, )
        expression = "({lhs})-({rhs})"
    return expression
    #]


def _default_item_iterator(value: Any, /, ) -> Iterator[Any]:
    r"""
    ................................................................................
    ==Default Item Iterator==

    Provides a default iterator for `Databox` items, ensuring compatibility with 
    both iterable and non-iterable data types. Non-iterables are wrapped in a list.

    ................................................................................

    ### Input arguments ###
    ???+ input "value"
        The input item to iterate over. Can be iterable or non-iterable.

    ### Returns ###
    ???+ returns "Iterator[Any]"
        An iterator over the item.

    ### Example ###
    ```python
        iterator = _default_item_iterator([1, 2, 3])
        for item in iterator:
            print(item)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Get Series Constructor==

    Returns a constructor function for creating `Series` objects based on a given 
    start date or a sequence of periods.

    ................................................................................

    ### Input arguments ###
    ???+ input "start"
        An optional `Period` object specifying the starting date for the series.

    ???+ input "periods"
        An optional iterable of `Period` objects defining the series range.

    ### Returns ###
    ???+ returns "Callable | None"
        A callable function for constructing `Series` objects, or `None` if both 
        inputs are `None`.

    ### Example ###
    ```python
        constructor = _get_series_constructor(start=Period("2023-01"))
        series = constructor(values=[1, 2, 3])
    ```
    ................................................................................
    """
    #[
    if start is not None:
        return lambda values: Series(start=start, values=values, )
    elif periods is not None:
        return lambda values: Series(periods=periods, values=values, )
    #]

