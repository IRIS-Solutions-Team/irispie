"""
Model equations
"""

#( External imports
from re import compile, Match
from typing import Self, Iterable
from dataclasses import dataclass
from enum import Flag, auto
from itertools import chain 
#)


#( Internal imports
from .incidence import Token, Tokens
from .exceptions import UndeclaredName
#)


_EVALUATOR_FORMAT = "lambda x, t: [{equations}]"
_X_REF_PATTERN = "{quantity_id},t{shift:+g},{equation_id}"

_QUANTITY_NAME_PATTERN = compile(r"\b([a-zA-Z]\w*)\b({[-+\d]+})?(?!\()")
_EQUATION_REF = "..."
_REPLACE_NAME = "x[" + _X_REF_PATTERN +"]"
_TRANSLATE_REF_TO_KEY = { "[": "['", "]": "]'" }
_REPLACE_UKNOWN = "?"



ErrorLogT = list[tuple[str, str]]



class EquationKind(Flag):
    #(
    UNSPECIFIED = auto()
    TRANSITION_EQUATION = auto()
    MEASUREMENT_EQUATION = auto()

    EQUATION = TRANSITION_EQUATION | MEASUREMENT_EQUATION
    FIRST_ORDER_SYSTEM = EQUATION
    #)



@dataclass
class Equation():
    id: int
    human: str
    kind: EquationKind
    xtring: str | None = None
    incidence: Tokens | None = None

    def finalize_from_human(self, name_to_id: dict[str, int]) -> tuple[Self, ErrorLogT]:
        _xtring, _incidence, error_log, *_ = _xtring_from_human(self.human, name_to_id)
        self.xtring = _xtring
        self.incidence = _incidence
        return self, error_log

    def replace_equation_ref_in_xtring(self, replacement):
        return self.xtring.replace(_EQUATION_REF, str(replacement))

    def remove_equation_ref_from_xtring(self):
        return self.xtring.replace(","+_EQUATION_REF, "")



def generate_all_tokens_from_equations(equations: Iterable[Equation]) -> Iterable[Token]:
    return chain.from_iterable(eqn.incidence for eqn in equations)



def finalize_equations_from_humans(
    equations: list[Equation],
    name_to_id: dict[str, int],
):
    error_log = []
    for eqn in equations:
        _, _error_log = eqn.finalize_from_human(name_to_id)
        error_log += _error_log
    if error_log:
        raise UndeclaredName(error_log)



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



def create_name_to_id_from_equations(equations: list[Equation]) -> dict[str, int]:
    """
    """
    all_names = sorted(list(set(generate_all_names(equations))))
    return { name: quantity_id for quantity_id, name in enumerate(all_names) }



def create_evaluator_func_string(equations: Iterable[str]) -> str:
    """
    """
    return _EVALUATOR_FORMAT.format(equations=" , ".join(equations))



def get_wrt_tokens_by_equations(
    equations: Iterable[Equation],
    wrt_tokens: Iterable[Token],
) -> dict[int, Iterable[Token]]:

    wrt_tokens_by_equations = {}
    for eqn in equations:
        wrt_tokens_by_equations[eqn.id] = [
            tok for tok in wrt_tokens
            if tok in eqn.incidence
        ]
    return wrt_tokens_by_equations



# Private


def _xtring_from_human( 
    human: str,
    name_to_id: dict[str, int],
) -> tuple[str, set[Token], ErrorLogT, list[Token]]:
    """
    Convert human string to xtring and retrieve incidence tokens
    """
    tokens_list: list[Token] = []
    error_log = []

    def _x_from_human(match: Match) -> str:
        name = match.group(1)
        quantity_id = name_to_id.get(name)
        if quantity_id is not None:
            shift = _resolve_shift_str(match.group(2))
            tokens_list.append(Token(quantity_id, shift))
            return _REPLACE_NAME.format(
                quantity_id=quantity_id,
                shift=shift,
                equation_id=_EQUATION_REF,
            )
        else:
            error_log.append((name, human))
            tokens_list.append(Token(None, None))
            return _REPLACE_UKNOWN

    xtring = _QUANTITY_NAME_PATTERN.sub(_x_from_human, human)
    xtring = _postprocess_xtring(xtring)
    return xtring, set(tokens_list), error_log, tokens_list



def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("{", "",).replace("}", "")) if shift_str is not None else 0


def _postprocess_xtring(equation: str) -> str:
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    lhs_rhs = equation.split("=")
    if len(lhs_rhs)==2:
        equation = lhs_rhs[0] + "-(" + lhs_rhs[1] + ")" 
    return equation


def generate_equation_ids_by_kind(
    equations: Iterable[Equation],
    kind: EquationKind,
) -> list[int]:
    return (eqn.id for eqn in equations if eqn.kind in kind)


