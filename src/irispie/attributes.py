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


class AttributesMixin:
    """
    """
    #[

    def has_attributes(self, *attributes) -> bool:
        attributes = (set((a, )) if isinstance(a, str) else set(a) for a in attributes)
        return all(self.attributes.intersection(a, ) for a in attributes)

    #]


def generate_by_attributes(
    with_attributes: Iterable[AttributesProtocol],
    *args,
) -> Iterable[AttributesProtocol]:
    """
    """
    return ( i for i in with_attributes if i.has_attributes(*args, ) )

