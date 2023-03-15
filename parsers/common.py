

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
KEYWORD_PREFIX = GRAMMAR["keyword_prefix"].literal
HUMAN_PREFIX = GRAMMAR["human_prefix"].literal


def compile_keywords_pattern(commands_list: Iterable[str]) -> re.Pattern:
    return re.compile(
        HUMAN_PREFIX + r"(" + "|".join(commands_list) + r")\b"
    )


def translate_keywords(commands_pattern: re.Pattern, code: str) -> str:
    return re.sub(commands_pattern, lambda m: KEYWORD_PREFIX + m.group(1), code)


def add_blank_lines(source: str) -> str:
    return "\n" + source + "\n"


def combine_source_files(
    source_files: str|Iterable[str], 
    /,
    joint: str = "\n\n",
) -> str:
    """
    """
    if isinstance(source_files, str):
        source_files = [source_files]
    source_string = []
    for f in source_files:
        with open(f, "r") as fid:
            source_string.append(fid.read())
    source_string = "\n\n".join(source_string)
    return source_string


