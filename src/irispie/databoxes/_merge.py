"""
Merge inlay
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
    MergeStrategyType = Literal[
        "stack", # Stack values as variants
        "hstack", # Legacy alias for stack, do not include in the docstring
        "replace", # Replace the existing value with the new one
        "discard", # Discard the new value and keep the existing one
        "silent", # Exactly the same as discard
        "warning", # Same as discard with a warning
        "error", # Throw an error for the first duplicate key
        "critical", # Throw a critical error for the first duplicate key
    ]

#]


class Inlay:
    """
    """
    #[

    @classmethod
    def merged(
        klass,
        databoxes: Iterable[Self],
        merge_strategy: MergeStrategyType = "stack",
    ) -> Self:
        """
        """
        out = klass()
        out.merge(databoxes, merge_strategy, )
        return out

    @_dm.reference(category="multiple", )
    def merge(
        self: Self,
        other: Self | Iterable[Self],
        merge_strategy: MergeStrategyType = "stack",
        # Do not include the following in the docstring
        action = None,
    ) -> None:
        r"""
................................................................................

==Merge Databoxes==

Combine one or more databoxes into a single databox using a specified merge
strategy to handle potential conflicts between duplicate keys.

    self.merge(
        other,
        merge_strategy="stack",
    )


### Input arguments ###

???+ input "other"
    The databox or iterable of databoxes to merge into the current databox. If
    merging a single databox, it should be passed directly; for multiple
    databoxes, pass an iterable containing all.

???+ input "merge_strategy"
    Determines how to process keys that exist in more than one databox. The
    default strategy is `"stack"`.

    * `"stack"`: Stack values; this means combine time series into multiple
    columns, or combine lists, or convert non-lists to lists for stacking.

    * `"replace"`: Replace existing values with new values.

    * `"discard"` and `"silent"`: Retain original values and ignore new values.

    * `"warning"`: Behave like `"discard"` but issue a warning for each conflict.

    * `"error"`: Raise an error on encountering the first duplicate key.

    * `"critical"`: Raise a critical error on encountering the first duplicate key.


### Returns ###

    This method modifies the databox in place and returns `None`.

................................................................................
        """
        # Legacy name
        if action is not None:
            _wa.warn("The 'action' input argument is deprecated; use 'merge_strategy' instead", )
            merge_strategy = action
        #
        merge_strategy_func = _MERGE_STRATEGY_DISPATCH[merge_strategy]
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
                    merge_strategy_func(self, key, value, stream, )
                else:
                    self[key] = value
        stream._raise()

    #]


def _merge_stack(
    self,
    key: str,
    value: Any,
    stream: _wrongdoings.Stream,
    /,
) -> None:
    """
    Horizontal stack of values: time series are concatenated, lists are
    extended, non-lists are converted to lists and extended.
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
    stream: _wrongdoings.Stream,
    /,
) -> None:
    """
    """
    #[
    self[key] = value
    #]


def _merge_discard(
    self,
    key: str,
    value: Any,
    stream: _wrongdoings.Stream,
    /,
) -> None:
    """
    """
    #[
    pass
    #]


def _merge_report(
    self,
    key: str,
    value: Any,
    stream: _wrongdoings.Stream,
    /,
) -> None:
    """
    """
    #[
    stream.add(key, )
    #]


_MERGE_STRATEGY_DISPATCH = {
    "stack": _merge_stack,
    "hstack": _merge_stack,
    "replace": _merge_replace,
    "discard": _merge_discard,
    "silent": _merge_report,
    "warning": _merge_report,
    "error": _merge_report,
    "critical": _merge_report,
}

