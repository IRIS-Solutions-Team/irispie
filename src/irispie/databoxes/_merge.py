"""
Merge mixin
"""


#[
from __future__ import annotations

from typing import (Self, Iterable, Literal, )
from .. import wrongdoings as _wrongdoings
from ..series import main as _series
from . import main as _databoxes
#]


class Inlay:
    """
    """
    #[

    def merge(
        self: Self,
        they: Iterable[Self] | Self,
        action: Literal["hstack", "replace", "discard", "silent", "warning", "error", "critical", ] = "hstack",
        **kwargs,
    ) -> None:
        """
        """
        action_func = _MERGE_ACTIONS[action]
        stream = _wrongdoings.create_stream(action, when_no_stream="silent", )(
            "Duplicate keys when merging databoxes",
        )
        if hasattr(they, "items", ):
            they = (they, )
        for t in they:
            for key, value in t.items():
                if key in self:
                    action_func(self, key, value, stream, **kwargs, )
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


_MERGE_ACTIONS = {
    "hstack": _merge_hstack,
    "replace": _merge_replace,
    "discard": _merge_discard,
    "silent": _merge_report,
    "warning": _merge_report,
    "error": _merge_report,
    "critical": _merge_report,
}

