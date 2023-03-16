"""
Model equations
"""

#[
from __future__ import annotations

from IPython import embed
import enum
import re
import dataclasses 
import itertools
import operator

from typing import Self, NoReturn
from collections.abc import Iterable

from .incidence import Token, Tokens, sort_tokens
from . import incidence
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
    STEADY_EVALUATOR = TRANSITION_EQUATION | MEASUREMENT_EQUATION
    #]


@dataclasses.dataclass
class Equation:
    """
    """
    id: int | None = None
    human: str | None = None
    kind: EquationKind = EquationKind.UNSPECIFIED
    descript: str | None = None
    xtring: str | None = None
    incidence: Tokens | None = None
    entry: int | None = None
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

    def set_id(self, qid: int) -> Self:
        self.id = qid
        return self
    #]


Equations: TypeAlias = Iterable[Equation]


STEADY_REF_PATTERN = re.compile(r"&x\b")


def generate_all_tokens_from_equations(equations: Equations) -> Iterable[Token]:
    return itertools.chain.from_iterable(eqn.incidence for eqn in equations)


def finalize_dynamic_equations(
    equations: Equations,
    name_to_id: dict[str, int],
) -> NoReturn:
    _finalize_equations_from_humans(equations, name_to_id)
    _replace_steady_ref(equations)


def finalize_steady_equations(
    equations: Equations,
    name_to_id: dict[str, int],
) -> NoReturn:
    _finalize_equations_from_humans(equations, name_to_id)
    _remove_steady_ref(equations)


def _replace_steady_ref(equations: Equations) -> NoReturn:
    for eqn in equations:
        eqn.xtring = re.sub(STEADY_REF_PATTERN, "L", eqn.xtring)


def _remove_steady_ref(equations: Equations) -> NoReturn:
    for eqn in equations:
        eqn.xtring = re.sub(STEADY_REF_PATTERN, "x", eqn.xtring)


def _finalize_equations_from_humans(
    equations: Equations,
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


def generate_all_names(equations: Equations) -> list[str]:
    """
    Extract all names from a list of equations
    """
    return generate_names_from_human(" ".join(e.human for e in equations))


def create_name_to_qid_from_equations(equations: Equations) -> dict[str, int]:
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
    all_wrt_tokens: WrtTokens|None =None,
) -> dict[int, WrtTokens]:
    """
    """
    #[
    eid_to_wrt_tokens = {}
    for eqn in equations:
        eid_to_wrt_tokens[eqn.id] = sort_tokens(
            wrt for wrt in all_wrt_tokens if wrt in eqn.incidence
        )
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
        equation = "-(" + lhs_rhs[0] + ")+" + lhs_rhs[1]
    return equation
    #]


def generate_eids_by_kind(
    equations: Equations,
    kind: EquationKind,
) -> Iterable[int]:
    return (eqn.id for eqn in equations if eqn.kind in kind)


def generate_equations_of_kind(
    equations: Equations,
    kind: EquationKind,
) -> Equations:
    return (eqn for eqn in equations if eqn.kind in kind)


def sort_equations(
    equations: Equations,
    /,
) -> Equations:
    return sorted(equations, key=operator.attrgetter("id"))


def get_min_shift_from_equations(
    equations: Equations,
    /,
) -> int:
    return incidence.get_min_shift(generate_all_tokens_from_equations(equations))


def get_max_shift_from_equations(
    equations: Equations,
    /,
) -> int:
    return incidence.get_max_shift(generate_all_tokens_from_equations(equations))

