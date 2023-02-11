

from __future__ import annotations

import re
import parsimonious
from collections.abc import Iterable


GRAMMAR_DEF = r"""

    keyword_prefix = "¡"
    human_prefix = "!"
    white_spaces = ~r"[ \t\n]*"
    non_keyword_prefixes = ~r"[^¡]+"

"""


GRAMMAR = parsimonious.grammar.Grammar(GRAMMAR_DEF)
_KEYWORD_PREFIX = GRAMMAR["keyword_prefix"].literal
_HUMAN_PREFIX = GRAMMAR["human_prefix"].literal


def compile_keywords_pattern(commands_list: Iterable[str]) -> re.Pattern:
    return re.compile(
        _HUMAN_PREFIX + r"(" + "|".join(commands_list) + r")\b"
    )


def translate_keywords(commands_pattern: re.Pattern, code: str) -> str:
    return re.sub(commands_pattern, lambda m: _KEYWORD_PREFIX + m.group(1), code)


