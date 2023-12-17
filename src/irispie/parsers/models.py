"""
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
import parsimonious as _pa
import functools as _ft
import re as _re

from . import common as _common
from . import _substitutions as _substitutions
#]


_WHERE_SUBSTITUTE = ["transition-equations", "measurement-equations"]


def from_string(source_string: str, /, ) -> sources.Source:
    """
    """
    source_string = _resolve_shortcut_keywords(source_string)
    source_string = _common.add_blank_lines(source_string)
    source_string = _translate_keywords(source_string)
    nodes = _GRAMMAR["source"].parse(source_string)
    parsed = _Visitor().visit(nodes)
    parsed = _substitutions.resolve_substitutions(parsed, _WHERE_SUBSTITUTE)
    return parsed


_GRAMMAR_DEF = _common.GRAMMAR_DEF + r"""

    source = block+
    block = white_spaces (qty_block / log_block / eqn_block) white_spaces

    keyword =
        transition_equations_keyword / measurement_equations_keyword
        / transition_variables_keyword
        / anticipated_shocks_keyword / unanticipated_shocks_keyword
        / measurement_variables_keyword / measurement_shocks_keyword
        / parameters_keyword / exogenous_variables_keyword
        / log_keyword / all_but_keyword
        / substitutions_keyword / autoswaps_simulate_keyword / autoswaps_steady_keyword
        / preprocessor_keyword / postprocessor_keyword

    description = (description_quote description_text description_quote)?
    description_quote = '"'
    description_text = ~r'[^"]*'

    block_attributes = "{" attribute_chain "}"
        attribute_chain = ~r"(\s*:\w+\s*)+"

    eqn_block = eqn_keyword block_attributes? eqn_ended* 
    eqn_keyword = transition_equations_keyword / measurement_equations_keyword
        / substitutions_keyword / autoswaps_simulate_keyword / autoswaps_steady_keyword
        / preprocessor_keyword / postprocessor_keyword
    eqn_ended = white_spaces description white_spaces eqn_body eqn_end
    eqn_end = ";" 

    eqn_body = eqn_version eqn_steady
    eqn_steady = (eqn_version_separator eqn_version)?
    eqn_version_separator = "!!"
    eqn_version = ~r"[\?\w\.,\(\)=:\+\^\-\*\{\}\[\]/ \t\n&\$]+"

    qty_block = qty_keyword block_attributes? qty_ended*
    qty_keyword =
        transition_variables_keyword
        / anticipated_shocks_keyword / unanticipated_shocks_keyword
        / measurement_variables_keyword / measurement_shocks_keyword
        / parameters_keyword / exogenous_variables_keyword
    qty_ended = white_spaces description white_spaces qty_name qty_end
    qty_name = ~r"\b[\?a-zA-Z]\w*\b"
    qty_end = ~r"[,;\s\n]+"

    log_block = log_keyword white_spaces all_but_flag white_spaces log_ended*
    all_but_flag = (all_but_keyword)?
    log_ended = white_spaces qty_name qty_end

    transition_variables_keyword = keyword_prefix "transition-variables"
    anticipated_shocks_keyword = keyword_prefix "anticipated-shocks"
    unanticipated_shocks_keyword = keyword_prefix "unanticipated-shocks"
    measurement_variables_keyword = keyword_prefix "measurement-variables"
    measurement_shocks_keyword = keyword_prefix "measurement-shocks"
    parameters_keyword = keyword_prefix "parameters"
    exogenous_variables_keyword = keyword_prefix "exogenous-variables"
    transition_equations_keyword = keyword_prefix "transition-equations"
    equations_keyword = keyword_prefix "equations"
    measurement_equations_keyword = keyword_prefix "measurement-equations"
    log_keyword = keyword_prefix "log-variables"
    all_but_keyword = keyword_prefix "all-but"
    substitutions_keyword = keyword_prefix "substitutions"
    autoswaps_simulate_keyword = keyword_prefix "autoswaps-simulate"
    autoswaps_steady_keyword = keyword_prefix "autoswaps-steady"
    preprocessor_keyword = keyword_prefix "preprocessor"
    postprocessor_keyword = keyword_prefix "postprocessor"

"""


_GRAMMAR = _pa.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = _common.compile_keywords_pattern(_KEYWORDS)
_translate_keywords = _ft.partial(_common.translate_keywords, _KEYWORDS_PATTERN)


class _Visitor(_pa.nodes.NodeVisitor):
    """
    """
    #[
    def __init__(self):
        super().__init__()
        self.content = {}

    def _add(self, block_name, new_content, ):
        if new_content:
            updated_content = self.content.setdefault(block_name, []) + new_content
            self.content[block_name] = updated_content

    def visit_source(self, node, visited_children):
        return self.content

    def _visit_block(self, node, visited_children):
        if not visited_children[2]:
            return
        block_name = visited_children[0]
        block_attributes = \
            (visited_children[1][0], ) \
            if visited_children[1] else (None, )
        block_content_with_attributes = [
            b + block_attributes
            for b in visited_children[2]
        ]
        self._add(block_name, block_content_with_attributes, )

    visit_qty_block = _visit_block
    visit_eqn_block = _visit_block

    def visit_block_attributes(self, node, visited_children):
        attributes = visited_children[1].strip().replace("(", "").replace(")", "")
        attributes = attributes.split(":")
        if len(attributes) < 2:
            return ()
        attributes = tuple(":" + a.strip() for a in attributes[1:])
        return attributes

    def visit_log_block(self, node, visited_children):
        block_name, _, _, _, block_content = visited_children
        self._add(block_name, block_content, )

    def visit_all_but_flag(self, node, visited_children):
        block_name = "all-but"
        flag = visited_children[0] if visited_children else ""
        self._add(block_name, [flag], )
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

    def _visit_abbreviated_keyword(self, node, visited_children):
        return "transition-" + visited_children[1]

    visit_transition_variables_keyword = _visit_keyword
    visit_anticipated_shocks_keyword = _visit_keyword
    visit_unanticipated_shocks_keyword = _visit_keyword
    visit_measurement_variables_keyword = _visit_keyword
    visit_measurement_shocks_keyword = _visit_keyword
    visit_parameters_keyword = _visit_keyword
    visit_exogenous_variables_keyword = _visit_keyword
    visit_transition_equations_keyword = _visit_keyword
    visit_equations_keyword = _visit_abbreviated_keyword
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
    ( _re.compile(_common.HUMAN_PREFIX + r"variables\b"), _common.HUMAN_PREFIX + r"transition-variables" ),
    ( _re.compile(_common.HUMAN_PREFIX + r"equations\b"), _common.HUMAN_PREFIX + r"transition-equations" ),
]

# ( _re.compile(_common.HUMAN_PREFIX + r"shocks\b"), _common.HUMAN_PREFIX + r"transition-shocks" ),


def _resolve_shortcut_keywords(source_string: str, /, ) -> str:
    for short, long in _SHORTCUT_KEYWORDS:
        source_string = _re.sub(short, long, source_string)
    return source_string


def _deblank(string: str, /, ) -> str:
    return "".join(string.split())

