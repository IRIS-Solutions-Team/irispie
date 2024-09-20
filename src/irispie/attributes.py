"""
"""


#[
from __future__ import absolute_import

from collections.abc import (Iterable, )
from typing import (Protocol, )
#]


class AttributesProtocol(Protocol, ):
    """
    """
    #[

    attributes: set[str]

    #]


def has_attributes(self, *attributes, ) -> bool:
    attributes = (set((a, )) if isinstance(a, str) else set(a) for a in attributes)
    return all(self.attributes.intersection(a, ) for a in attributes)


def generate_by_attributes(
    with_attributes: Iterable[AttributesProtocol],
    *args,
) -> Iterable[AttributesProtocol]:
    """
    """
    return ( i for i in with_attributes if i.has_attributes(*args, ) )


def serialize(attributes: set[str] | None, /, ) -> tuple[str] | None:
    """
    """
    return tuple(str(i) for i in attributes) if attributes is not None else None


def deserialize(data: tuple[str] | None, ) -> set[str]:
    """
    """
    return set(str(i) for i in data) if data is not None else None

