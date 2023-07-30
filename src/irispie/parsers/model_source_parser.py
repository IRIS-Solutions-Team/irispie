"""
"""


#[
from __future__ import annotations

import parsimonious as pa_
import functools as ft_
import re as re_
from collections.abc import (Iterable, )

from ..parsers import (common as co_, substitutions as su_, )
#]


_WHERE_SUBSTITUTE = ["transition-equations", "measurement-equations"]

def from_string(source_string: str, /, ) -> sources.Source:
    """
    """
    source_string = _resolve_shortcut_keywords(source_string)
    source_string = co_.add_blank_lines(source_string)
    source_string = _translate_keywords(source_string)
    nodes = _GRAMMAR["source"].parse(source_string)
    parsed = _Visitor().visit(nodes)
    parsed = su_.resolve_substitutions(parsed, _WHERE_SUBSTITUTE)
    return parsed


_GRAMMAR_DEF = co_.GRAMMAR_DEF + r"""

    source = block+
    block = white_spaces (qty_block / log_block / eqn_block) white_spaces

    keyword = 
        transition_equations_keyword / measurement_equations_keyword
        / transition_variables_keyword / transition_shocks_keyword / measurement_variables_keyword / measurement_shocks_keyword / parameters_keyword / exogenous_variables_keyword
        / log_keyword / all_but_keyword
        / substitutions_keyword / autoswaps_simulate_keyword / autoswaps_steady_keyword
        / preprocessor_keyword / postprocessor_keyword

    description = (description_quote description_text description_quote)?
    description_quote = '"'
    description_text = ~r'[^"]*'

    attributes = ~r"\([:\w,;\s]*\)"

    eqn_block = eqn_keyword attributes? eqn_ended* 
    eqn_keyword = transition_equations_keyword / measurement_equations_keyword
        / substitutions_keyword / autoswaps_simulate_keyword / autoswaps_steady_keyword
        / preprocessor_keyword / postprocessor_keyword
    eqn_ended = white_spaces description white_spaces eqn_body eqn_end
    eqn_end = ";" 

    eqn_body = eqn_version eqn_steady
    eqn_steady = (eqn_version_separator eqn_version)?
    eqn_version_separator = "!!"
    eqn_version = ~r"[\?\w\.,\(\)=:\+\^\-\*\{\}\[\]/ \t\n&\$]+"

    qty_block = qty_keyword attributes? qty_ended*
    qty_keyword = transition_variables_keyword / transition_shocks_keyword / measurement_variables_keyword / measurement_shocks_keyword / parameters_keyword / exogenous_variables_keyword
    qty_ended = white_spaces description white_spaces qty_name qty_end
    qty_name = ~r"\b[\?a-zA-Z]\w*\b"
    qty_end = ~r"[,;\s\n]+"

    log_block = log_keyword white_spaces all_but_flag white_spaces log_ended*
    all_but_flag = (all_but_keyword)?
    log_ended = white_spaces qty_name qty_end

    transition_variables_keyword = keyword_prefix "transition-variables"
    transition_shocks_keyword = keyword_prefix "transition-shocks"
    measurement_variables_keyword = keyword_prefix "measurement-variables"
    measurement_shocks_keyword = keyword_prefix "measurement-shocks"
    parameters_keyword = keyword_prefix "parameters"
    exogenous_variables_keyword = keyword_prefix "exogenous-variables"
    transition_equations_keyword = keyword_prefix "transition-equations"
    measurement_equations_keyword = keyword_prefix "measurement-equations"
    log_keyword = keyword_prefix "log-variables"
    all_but_keyword = keyword_prefix "all-but"
    substitutions_keyword = keyword_prefix "substitutions"
    autoswaps_simulate_keyword = keyword_prefix "autoswaps-simulate"
    autoswaps_steady_keyword = keyword_prefix "autoswaps-steady"
    preprocessor_keyword = keyword_prefix "preprocessor"
    postprocessor_keyword = keyword_prefix "postprocessor"

"""


_GRAMMAR = pa_.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = co_.compile_keywords_pattern(_KEYWORDS)
_translate_keywords = ft_.partial(co_.translate_keywords, _KEYWORDS_PATTERN)


class _Visitor(pa_.nodes.NodeVisitor):
    """
    """
    #[
    def __init__(self):
        super().__init__()
        self.content = {}

    def _add(self, block_name, new_content, attributes):
        # FIXME: handle attributes
        if new_content:
            updated_content = self.content.setdefault(block_name, []) + new_content
            self.content[block_name] = updated_content

    def visit_source(self, node, visited_children):
        return self.content

    def _visit_block(self, node, visited_children):
        block_name = visited_children[0]
        block_attributes = visited_children[1]
        block_content = visited_children[2]
        self._add(block_name, block_content, block_attributes)

    visit_qty_block = _visit_block
    visit_eqn_block = _visit_block

    def visit_attributes(self, node, visited_children):
        return node.text.replace("(", "").replace(")", "")

    def visit_log_block(self, node, visited_children):
        block_name, _, _, _, block_content = visited_children
        self._add(block_name, block_content, None)

    def visit_all_but_flag(self, node, visited_children):
        block_name = "all-but"
        flag = visited_children[0] if visited_children else ""
        self._add(block_name, [flag], None)
        return flag

    def visit_qty_ended(self, node, visited_children):
        _, description, _, name, _ = visited_children
        return (description, name)

    def visit_eqn_ended(self, node, visited_children):
        _, description, _, eqn, _ = visited_children
        return (description, eqn)

    def visit_log_ended(self, node, visited_children):
        _, name, _ = visited_children
        return name

    def _visit_keyword(self, node, visited_children):
        return visited_children[1]

    visit_transition_variables_keyword = _visit_keyword
    visit_transition_shocks_keyword = _visit_keyword
    visit_measurement_variables_keyword = _visit_keyword
    visit_measurement_shocks_keyword = _visit_keyword
    visit_parameters_keyword = _visit_keyword
    visit_exogenous_variables_keyword = _visit_keyword
    visit_transition_equations_keyword = _visit_keyword
    visit_measurement_equations_keyword = _visit_keyword
    visit_substitutions_keyword = _visit_keyword
    visit_autoswaps_simulate_keyword = _visit_keyword
    visit_autoswaps_steady_keyword = _visit_keyword
    visit_preprocessor_keyword = _visit_keyword
    visit_postprocessor_keyword = _visit_keyword
    visit_log_keyword = _visit_keyword
    visit_all_but_keyword = _visit_keyword

    def visit_qty_keyword(self, node, visited_children):
        return visited_children[0]

    def visit_eqn_keyword(self, node, visited_children):
        return visited_children[0]

    def visit_qty_name(self, node, visited_children):
        return node.text.strip()

    def visit_eqn_body(self, node, visited_children):
        dynamic = _deblank(visited_children[0])
        steady = _deblank(visited_children[1])
        return (dynamic, steady)

    def visit_eqn_steady(self, node, visited_children):
        return visited_children[0][1] if visited_children else ""

    def visit_description(self, node, visited_children):
        return visited_children[0][1] if visited_children else ""

    def visit_eqn_version(self, node, visited_children):
        return node.text.strip()

    def visit_description_text(self, node, visited_children):
        return node.text

    def generic_visit(self, node, visited_children):
        return visited_children or node.text or None
    #]


_SHORTCUT_KEYWORDS = [
    ( re_.compile(co_.HUMAN_PREFIX + r"variables\b"), co_.HUMAN_PREFIX + r"transition-variables" ),
    ( re_.compile(co_.HUMAN_PREFIX + r"shocks\b"), co_.HUMAN_PREFIX + r"transition-shocks" ),
    ( re_.compile(co_.HUMAN_PREFIX + r"equations\b"), co_.HUMAN_PREFIX + r"transition-equations" ),
]


def _resolve_shortcut_keywords(source_string: str, /, ) -> str:
    for short, long in _SHORTCUT_KEYWORDS:
        source_string = re_.sub(short, long, source_string)
    return source_string


def _deblank(string: str, /, ) -> str:
    return "".join(string.split())

