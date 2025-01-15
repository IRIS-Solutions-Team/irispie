"""
Merge mixin
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )
import warnings as _wa
import documark as _dm

from .. import wrongdoings as _wrongdoings
from ..series import main as _series
from . import main as _databoxes



if TYPE_CHECKING:
    from typing import (Self, Iterable, Literal, )
    MergeStrategyType = Literal["hstack", "replace", "discard", "silent", "warning", "error", "critical", ]
#]


class Inlay:
    r"""
................................................................................
==Inlay Class==

Represents a customizable container with merge functionality, supporting 
various strategies for handling duplicate keys when combining multiple 
databox-like objects. This class provides a framework for handling flexible 
merge operations with structured error reporting and conflict resolution.

It serves as a foundational tool in data management workflows, allowing for 
streamlined integration and manipulation of structured datasets.

................................................................................
    """
    #[

    @classmethod
    @_dm.reference(category="TBD", )
    def merged(
        klass,
        databoxes: Iterable[Self],
        merge_strategy: MergeStrategyType = "hstack",
    ) -> Self:
        r"""
................................................................................
==Class Method: Create Merged Instance==

Constructs a new instance of `Inlay` by merging multiple databoxes 
based on a specified strategy.

The resulting instance is populated by integrating the provided databoxes 
and applying the selected conflict resolution method.

................................................................................

### Input arguments ###
???+ input "databoxes"
    An iterable of `Inlay` objects to be merged.
    These objects will be combined using the specified merge strategy.

???+ input "merge_strategy"
    A `MergeStrategyType` indicating how to handle duplicate keys.
    Default is `"hstack"`.

### Returns ###
???+ returns "Self"
    A new `Inlay` instance populated with merged data.

### Example ###
```python
    merged_inlay = Inlay.merged([inlay1, inlay2], merge_strategy="replace")
    print(merged_inlay)
```
................................................................................
        """
        out = klass()
        out.merge(databoxes, merge_strategy, )
        return out

    @_dm.reference(category="TBD", )
    def merge(
        self: Self,
        other: Iterable[Self] | Self,
        merge_strategy: MergeStrategyType = "hstack",
        #
        action = None,
        **kwargs,
    ) -> None:
        r"""
................................................................................
==Merge Method==

Combines another databox or a collection of databoxes into the current instance.
Handles duplicate keys using a strategy specified by the caller.

This method is pivotal for data aggregation workflows, offering flexibility 
in how conflicts are resolved. A legacy parameter `action` is supported 
but deprecated in favor of `merge_strategy`.

................................................................................

### Input arguments ###
???+ input "other"
    A single `Inlay` instance or an iterable of such instances.
    Represents the databox(es) to be merged into the current instance.

???+ input "merge_strategy"
    A `MergeStrategyType` indicating the merge conflict resolution approach.
    Default is `"hstack"`.

???+ input "action"
    (Optional) A legacy input for specifying the merge strategy. Use of 
    this parameter is discouraged as it may be removed in future versions.

???+ input "kwargs"
    Additional parameters that can influence the behavior of the merge 
    strategy functions.

### Returns ###
???+ returns "None"
    The method modifies the current instance in place.

### Example ###
```python
    inlay.merge([inlay2, inlay3], merge_strategy="discard")
```
................................................................................
        """
        # Legacy name
        if action is not None:
            _wa.warn("The 'action' input argument is deprecated; use 'merge_strategy' instead", )
            merge_strategy = action
        #
        merge_strategy_func = _MERGE_STRATEGY[merge_strategy]
        stream = _wrongdoings.create_stream(
            merge_strategy,
            "Duplicate keys when merging databoxes",
            when_no_stream="silent",
        )
        if hasattr(other, "items", ):
            other = (other, )
        for t in other:
            for key, value in t.items():
                if key in self:
                    merge_strategy_func(self, key, value, stream, **kwargs, )
                else:
                    self[key] = value
        stream._raise()

    #]


def _merge_hstack(
    self,
    key: str,
    value: Any,
    /,
    *args,
) -> None:
    r"""
................................................................................
==Horizontal Stack Merge Strategy==

Implements the "hstack" merge strategy, which appends or combines values 
associated with duplicate keys into lists or series.

This strategy is suited for cases where data aggregation requires retaining 
all conflicting values instead of overwriting them.

................................................................................

### Input arguments ###
???+ input "self"
    The current `Inlay` instance being modified.

???+ input "key"
    A string representing the key under which the value resides.

???+ input "value"
    The value associated with the key to be merged.

???+ input "args"
    Additional arguments that are ignored in this strategy.

### Returns ###
???+ returns "None"
    The method modifies the current instance in place.

### Example ###
```python
    _merge_hstack(instance, "key1", [1, 2, 3])
```
................................................................................
    """
    #[
    if isinstance(value, _series.Series, ):
        self[key] = self[key] | value
        return
    if not isinstance(self[key], list):
        self[key] = [self[key], ]
    if not isinstance(value, list):
        value = [value, ]
    self[key] += value
    #]


def _merge_replace(
    self,
    key: str,
    value: Any,
    /,
    *args,
) -> None:
    r"""
................................................................................
==Replace Merge Strategy==

Implements the "replace" strategy, which overwrites existing values 
for duplicate keys with the new values provided.

This strategy is best suited for scenarios where the latest value is 
always considered the most accurate or relevant.

................................................................................

### Input arguments ###
???+ input "self"
    The current `Inlay` instance being modified.

???+ input "key"
    A string representing the key under which the value resides.

???+ input "value"
    The value associated with the key to replace the existing value.

???+ input "args"
    Additional arguments that are ignored in this strategy.

### Returns ###
???+ returns "None"
    The method modifies the current instance in place.

### Example ###
```python
    _merge_replace(instance, "key2", "new_value")
```
................................................................................
    """
    #[
    self[key] = value
    #]


def _merge_discard(
    self,
    key: str,
    value: Any,
    /,
    *args,
) -> None:
    r"""
................................................................................
==Discard Merge Strategy==

Implements the "discard" strategy, which ignores new values for duplicate keys 
and retains the original values in the current instance.

This strategy is useful when maintaining the integrity of the original 
data is a priority.

................................................................................

### Input arguments ###
???+ input "self"
    The current `Inlay` instance being modified.

???+ input "key"
    A string representing the key under which the value resides.

???+ input "value"
    The new value associated with the key that will be ignored.

???+ input "args"
    Additional arguments that are ignored in this strategy.

### Returns ###
???+ returns "None"
    The method does not alter the current instance.

### Example ###
```python
    _merge_discard(instance, "key3", "unused_value")
```
................................................................................
    """
    #[
    pass
    #]


def _merge_report(
    self,
    key: str,
    value: Any,
    stream,
    /,
    *args,
) -> None:
    r"""
................................................................................
==Report Merge Strategy==

Implements reporting strategies ("silent", "warning", "error", "critical") 
by logging duplicate keys to a stream.

The behavior depends on the stream configuration, ranging from silent 
logging to raising exceptions or critical errors.

................................................................................

### Input arguments ###
???+ input "self"
    The current `Inlay` instance being modified.

???+ input "key"
    A string representing the key under which the value resides.

???+ input "value"
    The value associated with the key causing the conflict.

???+ input "stream"
    A logging stream to record or handle duplicate key events.

???+ input "args"
    Additional arguments that are ignored in this strategy.

### Returns ###
???+ returns "None"
    The method logs conflicts without altering the current instance.

### Example ###
```python
    _merge_report(instance, "key4", "conflicting_value", stream)
```
................................................................................
    """
    #[
    stream.add(key, )
    #]


_MERGE_STRATEGY = {
    "hstack": _merge_hstack,
    "replace": _merge_replace,
    "discard": _merge_discard,
    "silent": _merge_report,
    "warning": _merge_report,
    "error": _merge_report,
    "critical": _merge_report,
}

