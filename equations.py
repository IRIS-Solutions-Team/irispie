"""
Model equations
"""

#[
from __future__ import annotations

import enum
import re
import dataclasses 
import itertools

from typing import Self, NoReturn
from collections.abc import Iterable

from .incidence import Token, Tokens, sort_tokens
from .exceptions import UndeclaredName
#]


_EVALUATOR_FORMAT = "lambda x, t: [{equations}]"
X_REF_PATTERN = "{qid},t{shift:+g},{eid}"

_QUANTITY_NAME_PATTERN = re.compile(r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()")
_EQUATION_REF = "..."
_REPLACE_NAME = "x[" + X_REF_PATTERN +"]"
_TRANSLATE_REF_TO_KEY = { "[": "['", "]": "]'" }
_REPLACE_UKNOWN = "?"


ErrorLogType = list[tuple[str, str]]


class EquationKind(enum.Flag):
    """
    Classification of model equations
    """
    #[
    UNSPECIFIED = enum.auto()
    TRANSITION_EQUATION = enum.auto()
    MEASUREMENT_EQUATION = enum.auto()

    EQUATION = TRANSITION_EQUATION | MEASUREMENT_EQUATION
    #]


@dataclasses.dataclass
class Equation:
    """
    """
    id: int | None = None
    human: str | None = None
    kind: EquationKind = EquationKind.UNSPECIFIED
    xtring: str | None = None
    incidence: Tokens | None = None
    """
    """
    #[
    def finalize_from_human(self, name_to_id: dict[str, int]) -> tuple[Self, ErrorLogType]:
        self.xtring, self.incidence, error_log, *_ = _xtring_from_human(self.human, name_to_id)
        return self, error_log

    def replace_equation_ref_in_xtring(self, replacement):
        return self.xtring.replace(_EQUATION_REF, str(replacement))

    def remove_equation_ref_from_xtring(self):
        return self.xtring.replace(","+_EQUATION_REF, "")
    #]


Equations: TypeAlias = Iterable[Equation]


def generate_all_tokens_from_equations(equations: Equations) -> Iterable[Token]:
    return itertools.chain.from_iterable(eqn.incidence for eqn in equations)


def finalize_equations_from_humans(
    equations: list[Equation],
    name_to_id: dict[str, int],
) -> NoReturn:
    """
    """
    #[
    error_log = []
    for eqn in equations:
        _, _error_log = eqn.finalize_from_human(name_to_id)
        error_log += _error_log
    if error_log:
        raise UndeclaredName(error_log)
    #]


def generate_names_from_human(human: str) -> Iterable[str]:
    """
    Generate all names from a single human string
    """
    return (f[0] for f in _QUANTITY_NAME_PATTERN.findall(human))


def generate_all_names(equations: list[Equation]) -> list[str]:
    """
    Extract all names from a list of equations
    """
    return generate_names_from_human(" ".join(e.human for e in equations))


def create_name_to_qid_from_equations(equations: list[Equation]) -> dict[str, int]:
    """
    """
    all_names = sorted(list(set(generate_all_names(equations))))
    return { name: qid for qid, name in enumerate(all_names) }


def create_evaluator_func_string(equations: Iterable[str]) -> str:
    """
    """
    return _EVALUATOR_FORMAT.format(equations=" , ".join(equations))


def create_eid_to_wrt_tokens(
    equations: Equations,
    wrt_tokens: Tokens|None =None,
) -> dict[int, Tokens]:
    """
    """
    #[
    eid_to_wrt_tokens = {}
    for eqn in equations:
        include_tokens = set(eqn.incidence)
        if wrt_tokens:
            include_tokens = include_tokens.intersection(wrt_tokens)
        eid_to_wrt_tokens[eqn.id] = sort_tokens(include_tokens)
    return eid_to_wrt_tokens
    #]


def _xtring_from_human( 
    human: str,
    name_to_id: dict[str, int],
) -> tuple[str, set[Token], ErrorLogType, Tokens]:
    """
    Convert human string to xtring and retrieve incidence tokens
    """
    #[
    tokens_list: list[Token] = []
    error_log = []

    def _x_from_human(match: re.Match) -> str:
        name = match.group(1)
        qid = name_to_id.get(name)
        if qid is not None:
            shift = _resolve_shift_str(match.group(2))
            tokens_list.append(Token(qid, shift))
            return _REPLACE_NAME.format(
                qid=qid,
                shift=shift,
                eid=_EQUATION_REF,
            )
        else:
            error_log.append((name, human))
            tokens_list.append(Token(None, None))
            return _REPLACE_UKNOWN

    xtring = _QUANTITY_NAME_PATTERN.sub(_x_from_human, human)
    xtring = _postprocess_xtring(xtring)
    return xtring, set(tokens_list), error_log, tokens_list
    #]


def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("{", "",).replace("}", "")) if shift_str is not None else 0


def _postprocess_xtring(equation: str) -> str:
    #[
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    lhs_rhs = equation.split("=")
    if len(lhs_rhs)==2:
        equation = lhs_rhs[0] + "-(" + lhs_rhs[1] + ")" 
    return equation
    #]


def generate_eids_by_kind(
    equations: Equations,
    kind: EquationKind,
) -> Iterable[int]:
    return (eqn.id for eqn in equations if eqn.kind in kind)

