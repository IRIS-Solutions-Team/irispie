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
    def _decorate(callable_obj, ) -> Callable:
        callable_obj._pages_reference = True
        for k, v in kwargs.items():
            setattr(callable_obj, f"_pages_{k}", v)
        if not hasattr(callable_obj, "_pages_category", ):
            callable_obj._pages_category = None
        if not hasattr(callable_obj, "_pages_call_name", ):
            callable_obj._pages_call_name = f"{callable_obj.__name__}"
        if not hasattr(callable_obj, "_pages_priority", ):
            callable_obj._pages_priority = _PRIORITY.get(callable_obj._pages_category, _PRIORITY[None], )
        if not hasattr(callable_obj, "_pages_add_heading", ):
            callable_obj._pages_add_heading = type(callable_obj, ) is not type
        if not hasattr(callable_obj, "_pages_call_name_is_code", ):
            callable_obj._pages_call_name_is_code = True
        if callable_obj._pages_add_heading:
            callable_obj.__doc__ = _add_heading(
                callable_obj.__doc__,
                callable_obj._pages_call_name,
                callable_obj._pages_call_name_is_code,
            )
        callable_obj._pages_tagline = _extract_tagline(callable_obj, )

        return callable_obj
    return _decorate


def delete_pages_attributes(callable_obj, ) -> None:
    """
    """
    for attr in dir(callable_obj):
        if attr.startswith("_pages_"):
            delattr(callable_obj, attr)


def _extract_tagline(callable_obj, ) -> str:
    m = _TAGLINE_PATTERN.search(callable_obj.__doc__, )
    return m.group(1, ) if m else ""


def _add_heading(docstring: str, call_name: str, call_name_is_code: bool, ) -> str:
    icon = "☐"
    heading_text = f"{icon} `{call_name}`" if call_name_is_code else call_name
    heading_decor = "\n" + "-"*len(heading_text) + "--\n"
    docstring = _remove_visual_dividers(docstring, )
    docstring = "\n" + heading_text + heading_decor + docstring
    docstring = _add_visual_dividers(docstring, )
    return docstring


_PRIORITY = {
    "constructor": 10,
    None: 0,
}


_DIVIDER_PATTERN = _re.compile(r"\n·{20,}\n", )
_DIVIDER_PATTERN_LEGACY = _re.compile(r"\n\.{20,}\n", )


def _remove_visual_dividers(docstring: str, ) -> str:
    docstring = _DIVIDER_PATTERN.sub("", docstring)
    docstring = _DIVIDER_PATTERN_LEGACY.sub("", docstring)
    return docstring


def _add_visual_dividers(docstring: str, ) -> str:
    divider = "\n" + "·"*80 + "\n"
    return divider + docstring + divider


