"""
"""


#[

from __future__ import annotations
from typing import Any

#]


_ContextType = dict[str, Any]
_PortableType = dict[str, None]


def to_portable(context: _ContextType) -> _PortableType:
    """
    """
    return {
        k: None
        for k in context.keys()
        if k != "__builtins__"
    }

def from_portable(portable: _PortableType) -> _ContextType:
    """
    """
    keys = tuple(portable.keys()) + ("__builtins__", )
    return { k: None for k in keys }

