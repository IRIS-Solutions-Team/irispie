"""
"""


from __future__ import annotations

from IPython import embed
import parsimonious
import functools
import re
from collections.abc import Iterable

from . import common


def from_string(source_string: str, /, ) -> sources.Source:
    """
    """
    source_string = _resolve_shortcut_keywords(source_string)
    source_string = common.add_blank_lines(source_string)
    source_string = _translate_keywords(source_string)
    nodes = _GRAMMAR["source"].parse(source_string)
    visitor = _Visitor()
    return visitor.visit(nodes)



_GRAMMAR_DEF = common.GRAMMAR_DEF + r"""

    source = block+
    block = white_spaces (qty_block / log_block / eqn_block) white_spaces

    keyword = 
        transition_equations_keyword / measurement_equations_keyword /
        transition_variables_keyword / transition_shocks_keyword / measurement_variables_keyword / measurement_shocks_keyword / parameters_keyword / exogenous_variables_keyword /
        log_keyword / all_but_keyword

    description = (description_quote description_text description_quote)?
    description_quote = '"'
    description_text = ~r'[^"]*'

    eqn_block = eqn_keyword eqn_ended* 
    eqn_keyword = transition_equations_keyword / measurement_equations_keyword
    eqn_ended = white_spaces description white_spaces eqn_body eqn_end
    eqn_end = ";" 

    eqn_body = eqn_version eqn_steady
    eqn_steady = (eqn_version_separator eqn_version)?
    eqn_version_separator = "!!"
    eqn_version = ~r"[\?\w\.\(\)=\+\^\-\*\{\}/ \t\n&\$]+"

    qty_block = qty_keyword qty_ended*
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

"""


_GRAMMAR = parsimonious.grammar.Grammar(_GRAMMAR_DEF)
_KEYWORDS = [ k.members[1].literal for k in _GRAMMAR["keyword"].members ]
_KEYWORDS_PATTERN = common.compile_keywords_pattern(_KEYWORDS)
_translate_keywords = functools.partial(common.translate_keywords, _KEYWORDS_PATTERN)


class _Visitor(parsimonious.nodes.NodeVisitor):
    """
    """
    #[
    def __init__(self):
        super().__init__()
        self.content = {}
        # self.source = sources.Source()

    def _add(self, block_name, new_content):
        if new_content:
            updated_content = self.content.setdefault(block_name, []) + new_content
            self.content[block_name] = updated_content

    def visit_source(self, node, visited_children):
        return self.content

    def _visit_block(self, node, visited_children):
        block_name = visited_children[0]
        block_content = visited_children[1]
        self._add(block_name, block_content)

    visit_qty_block = _visit_block
    visit_eqn_block = _visit_block

    def visit_log_block(self, node, visited_children):
        block_name, _, _, _, block_content = visited_children
        self._add(block_name, block_content)

    def visit_all_but_flag(self, node, visited_children):
        block_name = "all-but"
        flag = visited_children[0] if visited_children else ""
        self._add(block_name, [flag])
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
    visit_log_keyword = _visit_keyword
    visit_all_but_keyword = _visit_keyword

    def visit_qty_keyword(self, node, visited_children):
        return visited_children[0]

    def visit_eqn_keyword(self, node, visited_children):
        return visited_children[0]

    def visit_qty_name(self, node, visited_children):
        return node.text.strip()
 
    def visit_eqn_body(self, node, visited_children):
        dynamic = visited_children[0]
        steady = visited_children[1]
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
    ( re.compile(common.HUMAN_PREFIX + r"variables\b"), common.HUMAN_PREFIX + r"transition-variables" ),
    ( re.compile(common.HUMAN_PREFIX + r"shocks\b"), common.HUMAN_PREFIX + r"transition-shocks" ),
    ( re.compile(common.HUMAN_PREFIX + r"equations\b"), common.HUMAN_PREFIX + r"transition-equations" ),
]


def _resolve_shortcut_keywords(source_string: str, /, ) -> str:
    for short, long in _SHORTCUT_KEYWORDS:
        source_string = re.sub(short, long, source_string)
    return source_string


