"""
Resolve substitutions in model equations
"""


#[
from __future__ import annotations

import re as _re
import operator as _op
#]


def resolve_substitutions(
    parsed: dict[str, tuple],
    where_substitute: list[str],
    /,
) -> dict[str, tuple]:
    """
    """
    source = parsed.get("substitutions", None)
    if not source:
        return parsed
    definitions = _define_substitutions(source, )
    where_substitute = set(where_substitute).intersection(parsed.keys())
    subs_pattern = _re.compile(r"\$(" + "|".join(lhs for lhs in definitions.keys()) + r")\$")
    replace = lambda match: definitions[match.group(1)]
    make_substitutions = lambda source: _re.sub(subs_pattern, replace, source)
    for wh in where_substitute:
        parsed[wh] = [
            (label, (make_substitutions(dynamic), make_substitutions(steady)), attributes)
            for label, (dynamic, steady), attributes in parsed[wh]
        ]
    return parsed


def _define_substitutions(substitutions: list, /, ) -> dict[str, str]:
    return dict(_separate_lhs_rhs(s[1][0]) for s in substitutions)


_SUBS_NAME = _re.compile(r"\w+", )


def _separate_lhs_rhs(subs_string: str, /, ) -> tuple[str, str]:
    subs_string = subs_string.replace(":=", "=", )
    lhs, rhs = subs_string.split("=", maxsplit=1, )
    return lhs, rhs


