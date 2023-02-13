
from __future__ import annotations

import re
import parsimonious
import functools
import itertools
from collections.abc import Iterable
from typing import TypeAlias, Protocol

from . import common


_GRAMMAR_DEF = common.GRAMMAR_DEF + r"""

    source =  (control / non_keyword_prefixes)+
    control = white_spaces (for_do_block / if_then_block / end_keyword)

    keyword = (
        for_keyword / do_keyword / 
        if_keyword / then_keyword / elseif_keyword / else_keyword /
        end_keyword
    )

    for_do_block = for_keyword white_spaces for_control do_keyword
        for_control = for_control_name white_spaces "=" white_spaces for_token_ended*
        for_control_name = ~r"\?\w*"
        for_token_ended = for_token for_token_end
        for_token = ~r"\w+"
        for_token_end = ~r"[,;\s]+"
        for_keyword = keyword_prefix "for"
        do_keyword = keyword_prefix "do"


    if_then_block = if_keyword if_condition then_keyword
        if_condition = non_keyword_prefixes

        if_keyword = keyword_prefix "if"
        then_keyword = keyword_prefix "then"
        else_keyword = keyword_prefix "else"
        elseif_keyword = keyword_prefix "elseif"

    end_keyword = keyword_prefix "end"

"""


_GRAMMAR = parsimonious.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORD_PREFIX = _GRAMMAR["keyword_prefix"].literal
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = common.compile_keywords_pattern(_KEYWORDS)
_translate_keywords = functools.partial(common.translate_keywords, _KEYWORDS_PATTERN)


class _Visitor(parsimonious.nodes.NodeVisitor):
    """
    """
    #[
    def __init__(self, context=None):
        super().__init__()
        self.content = []
        self.context = context if context else {}

    def _add(self, new):
        self.content.append(new)

    def visit_source(self, node, visited_children):
        return self.content

    def visit_non_keyword_prefixes(self, node, visited_children):
        text = _strip_lines(node.text)
        if text:
            self._add(_Text(text))

    def visit_end_keyword(self, node, visited_children):
        self._add(_End())

    def visit_for_do_block(self, node, visited_children):
        control_name, tokens = visited_children[2]    
        self.content.append(_For(control_name, tokens))

    def visit_for_control(self, node, visited_children):
        return [visited_children[0], visited_children[4]]

    def visit_for_token_ended(self, node, visited_children):
        return visited_children[0]

    # def visit_for_token(self, node, visited_children):
    #    return node.text

    def visit_for_control_name_equals(self, node, visited_children):
        return visited_children[0][1] if visited_children else "?"

    # def visit_for_control_name(self, node, visited_children):
    #    return node.text

    def generic_visit(self, node, visited_children):
        return visited_children or node.text
    #]


class _ElementProtocol(Protocol):
    def resolve(self, sequence): ...
    def replace(self, pattern, replacement): ...


Sequence: TypeAlias = Iterable[_ElementProtocol]


class _Text:
    """
    """
    #[
    level = 0
    def __init__(self, content: str):
        self.content = content

    def replace(self, pattern: str, replacement: str):
        content = self.content.replace(pattern, replacement)
        return _Text(content)

    def resolve(self, sequence: Sequence):
        return self.content, sequence[1:]
    #]


class _For:
    """
    """
    #[
    level = 1
    def __init__(self, control_name, tokens):
        self.control_name = control_name
        self.tokens = tokens

    def replace(self, pattern, replacement):
        return self

    def _resolve_sequence(self, sequence, control_name, token):
        sequence = [ s.replace(control_name, token) for s in sequence ]
        return _resolve_sequence(sequence)

    def resolve(self, sequence):
        index_end = _find_matching_end(sequence)
        code = "\n".join(
            self._resolve_sequence(sequence[1:index_end], self.control_name, t)
            for t in self.tokens
        )
        return code, sequence[index_end+1:]
    #]


class _End:
    """
    """
    #[
    level = -1

    def replace(self, pattern, replacement):
        return self
    #]


def _find_matching_end(sequence: Sequence) -> int:
    level = list(itertools.accumulate(s.level for s in sequence))
    return level.index(0)


def _resolve_sequence(sequence: Sequence) -> str:
    code = ""
    while sequence:
        new_code, sequence = sequence[0].resolve(sequence)
        if new_code:
            code = code + "\n" + new_code if code else new_code
    return code


def _strip_lines(text: str) -> str:
    split_text = (t.strip() for t in text.split("\n"))
    return "\n".join(s for s in split_text if s)


_contextual_expression_pattern = re.compile(r"<([^>]*)>")


def _stringify(input):
    if isinstance(input, str):
        return input
    elif not isinstance(input, Iterable):
        return str(input)
    else:
        return ",".join([str(i) for i in input])


def _evaluate_contextual_expressions(text: str, context: dict, /):
    replace = lambda match: _stringify(eval(match.group(1), {}, context))
    return re.sub(_contextual_expression_pattern, replace, text)


def _is_preparser_needed(source: str, /) -> bool:
    return _KEYWORD_PREFIX in source


def from_string(
    source: str,
    context: dict | None = None,
    /
) -> tuple[str, dict]:
    """
    """
    info = {
        "context": None,
        "preparser_needed": None,
    }

    info["context"] = context if context else {}

    # Evaluate <...> expressions in the local context
    source = _evaluate_contextual_expressions(source, info["context"])

    # Replace human prefix (!) with processing prefix (¡)
    source = _translate_keywords(source)

    # Add blank lines pre and post source
    source = common.add_blank_lines(source)

    info["preparser_needed"] = _is_preparser_needed(source),

    if info["preparser_needed"]:
        # Parse input string structure into nodes
        # Visit nodes and create a sequence of elements
        # Resolve elements into preparsed string
        nodes = _GRAMMAR["source"].parse(source)
        visitor = _Visitor()
        sequence = visitor.visit(nodes)
        preparsed_source = _resolve_sequence(sequence)
    else:
        preparsed_source = source

    return preparsed_source, info

