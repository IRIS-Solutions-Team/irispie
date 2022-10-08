"""
Equation manipulator
"""

from __future__ import annotations
from re import sub, compile, Match
from typing import Optional, NamedTuple, Callable
from enum import Flag, auto
from functools import wraps
from collections.abc import Iterable


from .incidence import Incidence, Token


EQUATION_ID_PLACEHOLDER = "[_]"
EVALUATOR_FORMAT = "lambda x, t: [{equations}]"
_QUANTITY_NAME_PATTERN = compile(r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()")
_REPLACE_ZERO_SHIFT = "x[{quantity_id}][t]" + EQUATION_ID_PLACEHOLDER
_REPLACE_NONZERO_SHIFT = "x[{quantity_id}][t{shift:+g}]" + EQUATION_ID_PLACEHOLDER
_REPLACE_UKNOWN = "?"


ErrorLogT = list[tuple[str, str]]
XtringsT = list[str]


class EquationKind(Flag):
#(
    UNSPECIFIED = auto()
    TRANSITION_EQUATION = auto()
    MEASUREMENT_EQUATION = auto()

    EQUATION = TRANSITION_EQUATION | MEASUREMENT_EQUATION
    FIRST_ORDER_SYSTEM = EQUATION
#)



class Equation(NamedTuple):
    id: int
    human: str
    kind: EquationKind = EquationKind.UNSPECIFIED


EquationsT = list[Equation]


def xtrings_from_equations(
    equations: EquationsT,
    name_to_id: dict[str, int],
) -> tuple[XtringsT, set[Incidence]]:
#(
    xtrings: XtringsT = list()
    incidences: set[Incidence] = set()
    error_log: ErrorLogT = list()

    for eqn in equations:
        curr_xtring, curr_tokens, curr_error_log, *_ = xtring_from_human(eqn.human, name_to_id)
        incidences.update(Incidence(eqn.id, t) for t in curr_tokens)
        xtrings.append(curr_xtring) 
        error_log.extend(curr_error_log)

    if error_log:
        raise UndeclaredName(error_log)

    return xtrings, incidences
#)


def xtring_from_human( 
    human: str,
    name_to_id: dict[str, int],
) -> tuple[str, set[Token], ErrorLogT, list[Token]]:
#(
    tokens_list: list[Token] = []
    error_log: ErrorLogT = []

    def _replace_name_in_human(match: Match) -> str:
        name = match.group(1)
        quantity_id = name_to_id.get(name)
        if quantity_id is not None:
            shift = _resolve_shift_str(match.group(2))
            tokens_list.append(Token(quantity_id, shift))
            return (_REPLACE_NONZERO_SHIFT if shift!=0 else _REPLACE_ZERO_SHIFT).format(quantity_id=quantity_id, shift=shift)
        else:
            error_log.append((name, human))
            tokens_list.append(Token(None, None))
            return _REPLACE_UKNOWN

    xtring: str = _QUANTITY_NAME_PATTERN.sub(_replace_name_in_human, human)
    xtring = _postprocess(xtring)

    tokens_set = set(tokens_list)

    return xtring, tokens_set, error_log, tokens_list
#)


def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("{", "",).replace("}", "")) if shift_str is not None else 0


def _postprocess(equation: str) -> str:
#(
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    lhs_rhs = equation.split("=")
    if len(lhs_rhs)==2:
        equation = lhs_rhs[0] + "-(" + lhs_rhs[1] + ")" 
    return equation
#)


def names_from_human(human: str) -> list[str]:
    """
    Extract all names from a single human string
    """
    return list(set(f[0] for f in _QUANTITY_NAME_PATTERN.findall(human)))


def names_from_equations(equations: EquationsT) -> list[str]:
    """
    Extract all names from a list of Equation objects
    """
    return names_from_human(" ".join(e.human for e in equations))


def create_evaluator_func_string(equations: Iterable[str]) -> str:
    return EVALUATOR_FORMAT.format(equations=" , ".join(equations))

