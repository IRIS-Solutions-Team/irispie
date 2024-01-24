"""
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Any, NoReturn, TypeAlias, Protocol, )
import re as _re
import parsimonious as _pa
import functools as _ft
import itertools as _it
import copy as _cp

from .. import wrongdoings as _wrongdoings
from . import _pseudofunctions as _pseudofunctions
from . import common as _common
from . import _shifts as _shifts
from . import _lists as _lists
#]


def from_string(
    source: str,
    /,
    *,
    context: dict[str, Any] | None = None,
    save_preparsed: str = "",
) -> tuple[str, dict]:
    """
    """
    #[
    info = {
        "context": None,
        "preparser_needed": None,
    }

    info["context"] = context or {}

    # Remove NON-NESTED block comments #{ ... #} or %{ ... %}
    source = _remove_block_comments(source, )

    # Remove line comments %, #, ..., \
    source = _remove_comments(source, )

    # Evaluate <...> expressions in the local context
    source = _evaluate_contextual_expressions(source, info["context"])

    # Replace time shifts {...} by [...]
    source = _shifts.standardize_time_shifts(source)

    # Replace human prefix (!) with processing prefix (¡)
    source = _translate_keywords(source)

    # Add blank lines pre and post source
    source = _common.add_blank_lines(source)

    info["preparser_needed"] = _is_preparser_needed(source)

    preparsed_source = (
        _run_preparser_on_source_string(source, context, )
        if info["preparser_needed"] else source
    )

    # Expand lists
    preparsed_source = _lists.resolve_lists(preparsed_source, )

    # Resolve pseudofunctions: diff(...), ...
    preparsed_source = _pseudofunctions.resolve_pseudofunctions(preparsed_source, )

    if save_preparsed:
        _save_preparsed_source(preparsed_source, save_preparsed, )

    return preparsed_source, info
    #]


_GRAMMAR_DEF = _common.GRAMMAR_DEF + r"""

    source =  (control / text / list)+
    control = white_spaces (for_do_block / if_then_block / else_keyword / end_keyword)
    text = ~r"[^¡]+"

    keyword = (
        for_keyword / do_keyword / 
        if_keyword / then_keyword / else_keyword /
        end_keyword /
        list_keyword
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

    list = list_keyword "(" list_name ")"
    list_keyword = keyword_prefix "list"
    list_name = ~r"`\w+"

"""


_GRAMMAR = _pa.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = _common.compile_keywords_pattern(_KEYWORDS, )
_translate_keywords = _ft.partial(_common.translate_keywords, _KEYWORDS_PATTERN, )


class _Visitor(_pa.nodes.NodeVisitor):
    """
    """
    #[
    def __init__(self, context=None):
        super().__init__()
        self.content = []
        self.context = context or {}

    def _add(self, new):
        self.content.append(new)

    def visit_source(self, node, visited_children, ):
        return self.content

    def visit_end_keyword(self, node, visited_children, ):
        self._add(_End())

    def visit_else_keyword(self, node, visited_children, ):
        self._add(_Else())

    def visit_for_do_block(self, node, visited_children, ):
        control_name, tokens = visited_children[2]
        self._add(_For(control_name, tokens))

    def visit_for_control(self, node, visited_children, ):
        control_name = visited_children[0][0] if visited_children[0] else "?"
        tokens = visited_children[1]
        return [control_name, tokens]

    def visit_for_token_ended(self, node, visited_children, ):
        return visited_children[0]

    def visit_for_control_name_equals(self, node, visited_children, ):
        return visited_children[0]

    def visit_if_then_block(self, node, visited_children, ):
        condition = visited_children[1].strip()
        self._add(_If(condition))

    def visit_text(self, node, visited_children, ):
        text = _strip_lines(node.text)
        if text:
            self._add(_Text(text))

    def visit_list(self, node, visited_children, ):
        self._add(_Text(node.text))

    def generic_visit(self, node, visited_children, ):
        return visited_children or node.text
    #]


class _DirectiveProtocol(Protocol, ):
    def resolve(self, sequence, context, ): ...
    def replace(self, pattern, replacement, ): ...


Sequence: TypeAlias = Iterable[_DirectiveProtocol]


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
        self._control_name = control_name
        self._tokens = tokens

    def replace(self, pattern, replacement, /, ) -> Self:
        return self

    def _expand_tokens(self, for_body_sequence: Sequence, ):
        """
        """
        new = []
        for t in self._tokens:
            new.extend(s.replace(self._control_name, t) for s in _cp.deepcopy(for_body_sequence))
        return new

    def resolve(self, sequence: Sequence, context: dict, /, ):
        index_end = _find_matching_end(sequence)
        for_body_sequence = self._expand_tokens(sequence[1:index_end], )
        code = _resolve_sequence(for_body_sequence, context, )
        return code, sequence[index_end+1:]

    #]


class _If:
    """
    """
    #[
    level = 1

    def __init__(self, condition, /, ):
        self._condition_text = condition.strip()
        self._condition_result = None

    def replace(self, pattern, replacement) -> Self:
        self._condition_text = self._condition_text.replace(pattern, replacement)
        return self

    def resolve(self, sequence: Sequence, context: dict, /, ):
        """
        """
        self._evaluate_condition(context, )
        index_end = _find_matching_end(sequence, )
        index_else = _find_matching_else(sequence, ) or index_end
        if self._condition_result:
            code = _resolve_sequence(sequence[1:index_else], context, )
        else:
            code = _resolve_sequence(sequence[index_else+1:index_end], context, )
        return code, sequence[index_end+1:]

    def _evaluate_condition(
        self,
        context: dict[str, Any] | None,
        /,
    ) -> None:
        """
        """
        try:
            self._condition_result = bool(eval(self._condition_text, {}, context, ))
        except Exception as exc:
            raise _wrongdoings.IrisPieError(
                f"Failed to evaluate this !if condition: {self._condition_text}",
            ) from exc
    #]


class _Else:
    """
    """
    #[
    level = 0

    def replace(self, pattern, replacement):
        """
        """
        return self

    def resolve(self, *args, **kwargs, ) -> NoReturn:
        """
        """
        raise _wrongdoings.IrisPieError(
            "Misplaced preparsing directive !else",
        )
    #]


class _End:
    """
    """
    #[

    level = -1

    def replace(self, pattern, replacement):
        return self

    def resolve(self, *args, **kwargs, ) -> NoReturn:
        """
        """
        raise _wrongdoings.IrisPieError(
            "Misplaced preparsing directive !end",
        )

    #]


_LINE_COMMENT_PATTERN = _re.compile(r'"[^"\n]*"|[%#].*|\.\.\..*|\\.*')
_BLOCK_COMMENT_PATTERN = _re.compile(r"([%#]){.*?\1\}", _re.DOTALL)

def _remove_comments(source: str, /, ) -> str:
    return _re.sub(
        _LINE_COMMENT_PATTERN,
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        source,
    )

def _remove_block_comments(source: str, /, ) -> str:
    return _re.sub(_BLOCK_COMMENT_PATTERN, "", source, )

def _cumulate_level(sequence: Sequence, /, ) -> int:
    return list(_it.accumulate(s.level for s in sequence))


def _find_matching_end(sequence: Sequence, /, ) -> int:
    """
    Find !end that is on level 0
    """
    #[
    cum_level = _cumulate_level(sequence, )
    return cum_level.index(0)
    #]


def _find_matching_else(sequence: Sequence, /, ) -> int:
    """
    Find !else that is on level 0
    """
    #[
    cum_level = _cumulate_level(sequence, )
    return next(
        (i for i, l in enumerate(cum_level) if l == 1 and isinstance(sequence[i], _Else)),
        None,
    )
    #]


def _resolve_sequence(sequence: Sequence, context: dict, /, ) -> str:
    """
    """
    #[
    code = ""
    while sequence:
        new_code, sequence = sequence[0].resolve(sequence, context, )
        if code and new_code:
            code = code + "\n" + new_code
        elif new_code:
            code = new_code
        else:
            pass
    return code
    #]


def _strip_lines(text: str) -> str:
    split_text = (t.strip() for t in text.split("\n"))
    return "\n".join(s for s in split_text if s)


_CONTEXTUAL_EXPRESSION_PATTERN = _re.compile(r"<([^>]*)>")


def _stringify(input, /, ):
    if isinstance(input, str):
        return input
    elif not isinstance(input, Iterable):
        return str(input)
    else:
        return ",".join([_stringify(i, ) for i in input])


def _evaluate_contextual_expressions(text: str, context: dict, /):
    """
    """
    #[
    def _replace(match):
        try:
            expression = match.group(1).strip()
            return _stringify(eval(expression, {}, context, ), )
        except:
            raise Exception(f"Failed to evaluate this contextual expression: {expression}")
    return _re.sub(_CONTEXTUAL_EXPRESSION_PATTERN, _replace, text)
    #]


def _is_preparser_needed(source: str, /, ) -> bool:
    return _common.KEYWORD_PREFIX in source


def _save_preparsed_source(source: str, file_name: str, /, ) -> None:
    if file_name:
        with open(file_name, "wt+") as fid:
            fid.write(source)

def _run_preparser_on_source_string(source: str, context: dir, /, ) -> str:
    nodes = _GRAMMAR["source"].parse(source, )
    visitor = _Visitor()
    sequence = visitor.visit(nodes, )
    return _resolve_sequence(sequence, context, )


