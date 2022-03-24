
from __future__ import annotations

from re import sub, compile, Match
from typing import Optional

from .incidence import Token


DIFF_EQUATION_ID = "[_]"
_QUANTITY_NAME_PATTERN = compile(r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()")
_REPLACE_ZERO_SHIFT = "x[{quantity_id}][t]" + DIFF_EQUATION_ID
_REPLACE_NONZERO_SHIFT = "x[{quantity_id}][t{shift:+g}]" + DIFF_EQUATION_ID
_REPLACE_UKNOWN = "?"


def parse_equation(human: str, name_to_id: dict[str, int]) -> tuple[str, set[Token], list[tuple[str, str]], list[Token]]:
    #(
    tokens_list: list[Token] = []
    error_log: list[tuple[str, str]] = []

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

    equation: str = _QUANTITY_NAME_PATTERN.sub(_replace_name_in_human, human)
    equation = _postprocess(equation)

    tokens_set = set(tokens_list)

    return equation, tokens_set, error_log, tokens_list
    #)


def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("{", "",).replace("}", "")) if shift_str is not None else 0


def _postprocess(equation: str) -> str:
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    lhs_rhs = equation.split("=")
    if len(lhs_rhs)==2:
        equation = lhs_rhs[0] + "-(" + lhs_rhs[1] + ")" 
    return equation


def extract_names(equation: str) -> list[str]:
    return list(set(f[0] for f in _QUANTITY_NAME_PATTERN.findall(equation)))


