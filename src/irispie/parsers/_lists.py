"""
"""


#[
from __future__ import annotations

import re as _re

from . import common as _common
#]


_TYPE_PATTERN = _re.compile(r"(\w+)`(\w+)")
_LIST_PATTERN = _re.compile(_common.KEYWORD_PREFIX + r"list\(`(\w+)\)")


def resolve_lists(source: str, ) -> str:
    """
    """
    #[

    type_to_names = _create_type_to_names(source, )
    if type_to_names:
        source = _replace_lists(source, type_to_names, )
        source = _remove_types(source, )
    return source

    #]


def _create_type_to_names(source: str, ) -> dict[str, str]:
    """
    Collect all name`type and put then into a dict of {"type": {"name1", "name2", ...}}
    """
    #[

    type_to_names = dict()
    for match in _TYPE_PATTERN.finditer(source, ):
        name, _type = match.groups()
        if _type not in type_to_names:
            type_to_names[_type] = set()
        type_to_names[_type].add(name, )
    return type_to_names

    #]


def _replace_lists(
    source: str,
    type_to_names: dict[str, str],
) -> str:
    """
    Replace list(`type) with name1, name2, ...
    """
    #[

    def replace(match: _re.Match, ) -> str:
        _type, = match.groups()
        return ", ".join(type_to_names.get(_type, tuple(), ), )

    return _LIST_PATTERN.sub(replace, source, )
    #]


def _remove_types(source: str, ) -> str:
    """
    Replace name`type with name
    """
    #[

    def replace(match: _re.Match, ) -> str:
        name, *_ = match.groups()
        return name

    return _TYPE_PATTERN.sub(replace, source, )
    #]


