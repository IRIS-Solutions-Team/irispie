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
import jinja2 as _jj

from .. import wrongdoings as _wrongdoings
from . import _pseudofunctions as _pseudofunctions
from . import common as _common
from . import _shifts as _shifts
from . import _lists as _lists
#]


_JJ_ENVIRONMENT = _jj.Environment()


def from_string(
    source: str,
    *,
    context: dict[str, Any] | None = None,
    save_preparsed: str = "",
    jinja: bool = True,
) -> tuple[str, dict]:
    """
    """
    #[
    info = {
        "context": None,
        "preparser_needed": None,
    }

    context = (dict(context) if context else {}) | {"__builtins__": {}}
    info["context"] = context

    # Remove NON-NESTED block comments #{ ... #} or %{ ... %}
    source = _remove_block_comments(source, )

    # Run Jinja2 templater
    if jinja:
        source = _JJ_ENVIRONMENT.from_string(source, ).render(context, )

    # Remove line comments %, #, ..., \, except for #!, %!
    source = _remove_line_comments(source, )

    # Replace time shifts {...} by [...]
    source = _shifts.standardize_time_shifts(source)

    # Replace human prefix (!) with processing prefix (¡)
    source = _translate_keywords(source)

    # Add blank lines pre and post source
    source = _common.add_blank_lines(source)

    # Run preparser on !-commands only if necessary
    preparser_needed = _is_preparser_needed(source)
    preparsed_source = (
        _run_preparser_on_source_string(source, context, )
        if preparser_needed else source
    )

    # Remove multiple blank lines
    preparsed_source = _consolidate_lines(preparsed_source, )

    # Evaluate and stringify Python expressions in <...> or <<...>>
    # Do this only after the !-command preparser because the <...> expression
    # may contain ?(...) control tokens
    preparsed_source = _evaluate_contextual_expressions(preparsed_source, info["context"], )

    # Expand lists
    preparsed_source = _lists.resolve_lists(preparsed_source, )

    # Resolve pseudofunctions: diff(...), ...
    preparsed_source = _pseudofunctions.resolve_pseudofunctions(preparsed_source, )

    if save_preparsed:
        _save_preparsed_source(preparsed_source, save_preparsed, )

    info = {
        "preparser_needed": preparser_needed,
        "preparsed_source": preparsed_source,
    }

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
        for_control = for_control_name_equals? for_tokens
        for_control_name_equals = for_control_name white_spaces ~r"[=:]" white_spaces
        for_control_name = ~r"\?[\w\(\)]*"
        for_tokens = ~r"\s*[^¡]+\s*"
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
        self.context = (dict(context) if context else {}) | {"__builtins__": {}}

    def _add(self, new):
        self.content.append(new)

    def visit_source(self, node, visited_children, ):
        return self.content

    def visit_end_keyword(self, node, visited_children, ):
        self._add(_End())

    def visit_else_keyword(self, node, visited_children, ):
        self._add(_Else())

    def visit_for_do_block(self, node, visited_children, ):
        control_name, tokens_as_string = visited_children[2]
        self._add(_For(control_name, tokens_as_string))

    def visit_for_control(self, node, visited_children, ):
        control_name = visited_children[0][0] if visited_children[0] else "?"
        tokens = visited_children[1]
        return [control_name, tokens]

    def visit_for_tokens(self, node, visited_children, ):
        return node.text

    def visit_for_control_name_equals(self, node, visited_children, ):
        return visited_children[0]

    def visit_if_then_block(self, node, visited_children, ):
        condition = visited_children[1].strip()
        self._add(_If(condition))

    def visit_text(self, node, visited_children, ):
        text = node.text
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

    def __init__(self, control_name, tokens_as_string, /, ):
        self._control_name = control_name
        self._tokens_as_string = tokens_as_string

    def replace(self, pattern, replacement, /, ) -> Self:
        self._tokens_as_string = self._tokens_as_string.replace(pattern, replacement, )
        return self

    def _expand_tokens(self, for_body_sequence: Sequence, context: dict, /, ):
        """
        """
        tokens = self._prepare_tokens(context, )
        control_name = self._control_name
        bare_control_name = control_name.removeprefix("?")
        #
        if bare_control_name.startswith("(") and bare_control_name.endswith(")"):
            upper_control_name = control_name.replace("(", "{", ).replace(")", "}", )
            lower_control_name = control_name.replace("(", "[", ).replace(")", "]", )
            control_name_upper = control_name + "|upper"
            control_name_lower = control_name + "|lower"
            def replace_control_by_token(source_text, token):
                token_upper = token.upper()
                token_lower = token.lower()
                source_text = source_text.replace(upper_control_name, token_upper, )
                source_text = source_text.replace(lower_control_name, token_lower, )
                source_text = source_text.replace(control_name_upper, token_upper, )
                source_text = source_text.replace(control_name_lower, token_lower, )
                source_text = source_text.replace(control_name, token, )
                return source_text
        else:
            def replace_control_by_token(source_text, token):
                source_text = source_text.replace(control_name, token, )
                return source_text
        #
        new = []
        for t in tokens:
            new.extend(
                replace_control_by_token(s, t, )
                for s in _cp.deepcopy(for_body_sequence, )
            )
        return new

    def _prepare_tokens(self, context: dict, ):
        """
        """
        tokens_as_string = _evaluate_contextual_expressions(self._tokens_as_string, context, )
        return _re.findall(r"\w+", tokens_as_string, )

    def resolve(self, sequence: Sequence, context: dict, /, ):
        """
        """
        index_end = _find_matching_end(sequence)
        for_body_sequence = self._expand_tokens(sequence[1:index_end], context, )
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
        context = (dict(context) if context else {}) | {"__builtins__": {}}
        try:
            value = eval(self._condition_text, context, )
            self._condition_result = bool(value)
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


_LINE_COMMENT_PATTERN = _re.compile(r'"[^"\n]*"|[%#](?!!).*|\.\.\..*|\\.*', )
_BLOCK_COMMENT_PATTERN = _re.compile(r"([%#]){.*?\1\}", _re.DOTALL, )


def _remove_line_comments(source: str, /, ) -> str:
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


def _consolidate_lines(text: str) -> str:
    split_text = _re.split(" *\n+", text, )
    return "\n".join(i for i in split_text if i)


_CONTEXTUAL_EXPRESSION_PATTERN = _re.compile(r"<+([^>]*)>+")


def _evaluate_contextual_expressions(
    text: str,
    context: dict[str, Any] | None,
) -> str:
    """
    """
    #[
    context = (dict(context, ) if context else {}) | {"__builtins__": {}}
    def _replace(match, ):
        expression = match.group(1).strip()
        try:
            value = eval(expression, context, )
            return _stringify(value, )
        except:
            raise Exception(f"Failed to evaluate this contextual expression: {expression}")
    return _re.sub(_CONTEXTUAL_EXPRESSION_PATTERN, _replace, text)
    #]


def _stringify(input, /, ):
    if isinstance(input, str):
        return input
    elif not isinstance(input, Iterable):
        return str(input)
    else:
        return ",".join(_stringify(i, ) for i in input)


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


