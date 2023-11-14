"""
Utility iterators
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, Iterator, )
from typing import (Any, )
#]


def exhaust_then_last(
    iterable: Iterable[Any],
    default=None,
    /,
) -> Iterator[Any]:
    """
    Repeat the last element of an iterable indefinitely
    """
    last = default
    for item in iterable:
        yield item
        last = item
    while True:
        yield last

