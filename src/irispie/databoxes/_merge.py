"""
Merge mixin
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )
import warnings as _wa

from .. import wrongdoings as _wrongdoings
from ..series import main as _series
from . import main as _databoxes

if TYPE_CHECKING:
    from typing import (Self, Iterable, Literal, )
    MergeStrategyType = Literal["hstack", "replace", "discard", "silent", "warning", "error", "critical", ]
#]


class Inlay:
    """
    """
    #[

    @classmethod
    def merged(
        klass,
        databoxes: Iterable[Self],
        merge_strategy: MergeStrategyType = "hstack",
    ) -> Self:
        """
        """
        out = klass()
        out.merge(databoxes, merge_strategy, )
        return out

    def merge(
        self: Self,
        other: Iterable[Self] | Self,
        merge_strategy: MergeStrategyType = "hstack",
        #
        action = None,
        **kwargs,
    ) -> None:
        """
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
    """
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
    """
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
    """
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
    """
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

