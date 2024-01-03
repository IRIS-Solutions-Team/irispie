"""
Documentation introspection tools
"""


#[
from __future__ import annotations

from collections.abc import (Callable, )
import re as _re
import os as _os
#]


_TAGLINE_PATTERN = _re.compile(r"==(.*?)==", )


def reference(**kwargs, ) -> Callable:
    def _decorate(callable_object, ) -> Callable:
        callable_object._pages_reference = True
        for k, v in kwargs.items():
            setattr(callable_object, f"_pages_{k}", v)
        if not hasattr(callable_object, "_pages_call_name", ):
            callable_object._pages_call_name = callable_object.__name__
        if not hasattr(callable_object, "_pages_category", ):
            callable_object._pages_category = None
        callable_object._pages_tagline = _extract_tagline(callable_object, )

        return callable_object
    return _decorate


def _extract_tagline(callable_object, ) -> str:
    m = _TAGLINE_PATTERN.search(callable_object.__doc__, )
    return m.group(1, ) if m else ""

