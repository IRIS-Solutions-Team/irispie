"""
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
import re as _re
import parsimonious as _pa
#]


GRAMMAR_DEF = r"""

    keyword_prefix = "¡"
    human_prefix = "!"
    white_spaces = ~r"[ \t\n]*"
    non_keyword_prefixes = ~r"[^¡]+"

"""


GRAMMAR = _pa.grammar.Grammar(GRAMMAR_DEF)
KEYWORD_PREFIX = GRAMMAR["keyword_prefix"].literal
HUMAN_PREFIX = GRAMMAR["human_prefix"].literal


def compile_keywords_pattern(commands_list: Iterable[str]) -> _re.Pattern:
    return _re.compile(
        HUMAN_PREFIX + r"(" + "|".join(commands_list) + r")\b"
    )


def translate_keywords(commands_pattern: _re.Pattern, code: str) -> str:
    return _re.sub(commands_pattern, lambda m: KEYWORD_PREFIX + m.group(1), code)


def add_blank_lines(source: str) -> str:
    return "\n" + source + "\n"


