"""
"""


#[
from __future__ import annotations
from IPython import embed

import re as re_
import parsimonious
import functools
import itertools
from collections.abc import Iterable
from typing import (TypeAlias, Protocol, )

from ..parsers import (common as co_, pseudofunctions as pf_, shifts as sh_, )
#]


def from_string(
    source: str,
    context: dict | None = None,
    /,
    save_preparsed: str = "",
) -> tuple[str, dict]:
    """
    """
    info = {
        "context": None,
        "preparser_needed": None,
    }

    info["context"] = context if context else {}

    # Remove line comments %, #, ...
    source = _remove_comments(source)

    # Evaluate <...> expressions in the local context
    source = _evaluate_contextual_expressions(source, info["context"])

    # Replace time shifts {...} by [...]
    source = sh_.standardize_time_shifts(source)

    # Replace human prefix (!) with processing prefix (¡)
    source = _translate_keywords(source)

    # Add blank lines pre and post source
    source = co_.add_blank_lines(source)

    info["preparser_needed"] = _is_preparser_needed(source)

    preparsed_source = _run_preparser_on_source_string(source, context, ) if info["preparser_needed"] else source

    # Resolve pseudofunctions
    preparsed_source = pf_.resolve_pseudofunctions(preparsed_source)

    if save_preparsed:
        _save_preparsed_source(preparsed_source, save_preparsed)

    return preparsed_source, info


_GRAMMAR_DEF = co_.GRAMMAR_DEF + r"""

    source =  (control / text)+
    control = white_spaces (for_do_block / if_then_block / else_keyword / end_keyword)
    text = ~r"[^¡]+"

    keyword = (
        for_keyword / do_keyword / 
        if_keyword / then_keyword / else_keyword /
        end_keyword
    )

    for_do_block = for_keyword white_spaces for_control do_keyword
        for_control = for_control_name_equals? for_token_ended*
        for_control_name_equals = for_control_name white_spaces "=" white_spaces
        for_control_name = ~r"\?\w*"
        for_token_ended = for_token for_token_end
        for_token = ~r"\w+"
        for_token_end = ~r"[,;\s]+"
        for_keyword = keyword_prefix "for"
        do_keyword = keyword_prefix "do"

    if_then_block = if_keyword if_condition then_keyword
        if_condition = ~r"[^¡]+"
        if_keyword = keyword_prefix "if"
        then_keyword = keyword_prefix "then"
        else_keyword = keyword_prefix "else"

    end_keyword = keyword_prefix "end"

"""


_GRAMMAR = parsimonious.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORD_PREFIX = _GRAMMAR["keyword_prefix"].literal
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = co_.compile_keywords_pattern(_KEYWORDS)
_translate_keywords = functools.partial(co_.translate_keywords, _KEYWORDS_PATTERN)


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

    def visit_end_keyword(self, node, visited_children):
        self._add(_End())

    def visit_for_do_block(self, node, visited_children):
        control_name, tokens = visited_children[2]
        self._add(_For(control_name, tokens))

    def visit_for_control(self, node, visited_children):
        control_name = visited_children[0][0] if visited_children[0] else "?"
        tokens = visited_children[1]
        return [control_name, tokens]

    def visit_for_token_ended(self, node, visited_children):
        return visited_children[0]

    def visit_for_control_name_equals(self, node, visited_children):
        return visited_children[0]

    def visit_if_then_block(self, node, visited_children):
        condition = visited_children[1].strip()
        self._add(_If(condition))

    def visit_text(self, node, visited_children):
        text = _strip_lines(node.text)
        if text:
            self._add(_Text(text))

    def generic_visit(self, node, visited_children):
        return visited_children or node.text
    #]


class _ElementProtocol(Protocol, ):
    def resolve(self, sequence, context, ): ...
    def replace(self, pattern, replacement, ): ...


Sequence: TypeAlias = Iterable[_ElementProtocol]


class _Text:
    """
    """
    #[
    level = 0
    def __init__(self, content: str, ):
        self.content = content

    def replace(self, pattern: str, replacement: str, ):
        new = _Text(self.content.replace(pattern, replacement))
        return new

    def resolve(self, sequence: Sequence, context: dict, ):
        return self.content, sequence[1:]
    #]


class _For:
    """
    """
    #[
    level = 1
    def __init__(self, control_name, tokens, /, ):
        self.control_name = control_name
        self.tokens = tokens

    def replace(self, pattern, replacement, /, ):
        return self

    def _resolve_sequence(self, sequence, context, control_name, token, /, ):
        sequence = [ s.replace(control_name, token) for s in sequence ]
        return _resolve_sequence(sequence, context)

    def resolve(self, sequence: Sequence, context: dict, /, ):
        index_end = _find_matching_end(sequence)
        code = "\n".join(
            self._resolve_sequence(sequence[1:index_end], context, self.control_name, t)
            for t in self.tokens
        )
        return code, sequence[index_end+1:]
    #]


class _If:
    """
    """
    #[
    level = 1

    def __init__(self, condition, /, ):
        self.condition_text = condition
        self.condition_result = None

    def replace(self, pattern, replacement):
        self.condition_text = self.condition_text.replace(pattern, replacement)
        return self

    def resolve(self, sequence: Sequence, context: dict, /, ):
        raise Exception("!if not implemented yet")
        index_end = _find_matching_end(sequence)
        code = ...
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


_COMMENTS_PATTERN = re_.compile(r'"[^"\n]*"|[%#].*|\.\.\..*')


def _remove_comments(source: str, /, ) -> str:
    return re_.sub(
        _COMMENTS_PATTERN,
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        source,
    )


def _find_matching_end(sequence: Sequence, /, ) -> int:
    level = list(itertools.accumulate(s.level for s in sequence))
    return level.index(0)


def _resolve_sequence(sequence: Sequence, context: dict, /, ) -> str:
    code = ""
    while sequence:
        new_code, sequence = sequence[0].resolve(sequence, context, )
        if new_code:
            code = code + "\n" + new_code if code else new_code
    return code


def _strip_lines(text: str) -> str:
    split_text = (t.strip() for t in text.split("\n"))
    return "\n".join(s for s in split_text if s)


_contextual_expression_pattern = re_.compile(r"<([^>]*)>")


def _stringify(input, /, ):
    if isinstance(input, str):
        return input
    elif not isinstance(input, Iterable):
        return str(input)
    else:
        return ",".join([_stringify(i, ) for i in input])

def _evaluate_contextual_expressions(text: str, context: dict, /):
    def _replace(match):
        try:
            return _stringify(eval(match.group(1), {}, context), )
        except:
            raise Exception(f"Error evaluating this contextual expresssion: {match.group(0)}")
    return re_.sub(_contextual_expression_pattern, _replace, text)


def _is_preparser_needed(source: str, /, ) -> bool:
    return _KEYWORD_PREFIX in source


def _save_preparsed_source(source: str, file_name: str, /, ) -> NoReturn:
    if file_name:
        with open(file_name, "wt+") as fid:
            fid.write(source)

def _run_preparser_on_source_string(source: str, context: dir, /, ) -> str:
    nodes = _GRAMMAR["source"].parse(source, )
    visitor = _Visitor()
    sequence = visitor.visit(nodes, )
    return _resolve_sequence(sequence, context, )


