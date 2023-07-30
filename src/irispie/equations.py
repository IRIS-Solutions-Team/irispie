"""
Model equations
"""

#[
import enum as _en
import re as _re
import dataclasses as _dc
import itertools as _it
import operator as _op
from typing import (Self, TypeAlias, )
from collections.abc import (Iterable, )

from . import (quantities as _qu, incidence as _in, wrongdoings as _wd, )
#]


EVALUATOR_PREAMBLE = "lambda x, t, L: "
_EVALUATOR_FORMAT = EVALUATOR_PREAMBLE + "[{joined_xtrings}]"
X_REF_PATTERN = "{qid},t{shift:+g},{eid}"

_QUANTITY_NAME_PATTERN = _re.compile(r"\b([a-zA-Z]\w*)\b(\[[-+\d]+\])?(?!\()")
_EQUATION_REF = "..."
_REPLACE_NAME = "x[" + X_REF_PATTERN +"]"
_TRANSLATE_REF_TO_KEY = {
    "[": "['",
    "]": "]'",
}
_REPLACE_UKNOWN = "?"


ErrorLogType = list[tuple[str, str]]


class EquationKind(_en.Flag):
    """
    Classification of model equations
    """
    #[
    TRANSITION_EQUATION = _en.auto()
    MEASUREMENT_EQUATION = _en.auto()
    #]


__all__  = [
    "TRANSITION_EQUATION", "MEASUREMENT_EQUATION",
]
for n in __all__:
    exec(f"{n} = EquationKind.{n}")


@_dc.dataclass(slots=True, )
class Equation:
    """
    """
    id: int | None = None
    human: str | None = None
    kind: EquationKind | None = None
    description: str | None = None
    xtring: str | None = None
    incidence: _in.Tokens | None = None
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

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__)
    #]


Equations: TypeAlias = Iterable[Equation]


def generate_all_tokens_from_equations(equations: Equations) -> _in.Tokens:
    return _it.chain.from_iterable(eqn.incidence for eqn in equations)


def finalize_dynamic_equations(
    equations: Equations,
    name_to_id: dict[str, int],
) -> None:
    _finalize_equations_from_humans(equations, name_to_id)
    _replace_steady_ref(equations)


def finalize_steady_equations(
    equations: Equations,
    name_to_id: dict[str, int],
) -> None:
    _finalize_equations_from_humans(equations, name_to_id)
    _remove_steady_ref(equations)


def _replace_steady_ref(equations: Equations) -> None:
    STEADY_REF_PATTERN = _re.compile(r"&x\[([^,\]]+),[^\]]+\]")
    for eqn in equations:
        eqn.xtring = _re.sub(STEADY_REF_PATTERN, lambda match: "L["+match.group(1)+"]", eqn.xtring)


def _remove_steady_ref(equations: Equations) -> None:
    STEADY_REF_PATTERN = _re.compile(r"&x\b")
    for eqn in equations:
        eqn.xtring = _re.sub(STEADY_REF_PATTERN, "x", eqn.xtring)


def _finalize_equations_from_humans(
    equations: Equations,
    name_to_id: dict[str, int],
) -> None:
    """
    """
    #[
    error_log = []
    for eqn in equations:
        _, _error_log = eqn.finalize_from_human(name_to_id)
        error_log += _error_log
    if error_log:
        raise _wd.IrisPieError(
            ["Some names used in equations not declared"]
            + error_log
        )
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


def create_equator_func_string(xtrings: str) -> str:
    """
    """
    return _EVALUATOR_FORMAT.format(joined_xtrings=" , ".join(xtrings))


def create_eid_to_wrt_tokens(
    equations: Equations,
    all_wrt_tokens: _in.Tokens,
) -> dict[int, _in.Tokens]:
    """
    """
    #[
    eid_to_wrt_tokens = {}
    for eqn in equations:
        eid_to_wrt_tokens[eqn.id] = _in.sort_tokens(
            wrt for wrt in all_wrt_tokens
            if wrt in eqn.incidence
        )
    return eid_to_wrt_tokens
    #]


def create_human_to_eid(
    equations: Equations,
    /,
) -> dict[str, int]:
    return { eqn.human: eqn.id for eqn in equations }


def _xtring_from_human( 
    human: str,
    name_to_id: dict[str, int],
) -> tuple[str, set[_in.Token], ErrorLogType, _in.Tokens]:
    """
    Convert human string to xtring and retrieve incidence tokens
    """
    #[
    tokens_list: list[_in.Token] = []
    error_log = []

    def _x_from_human(match: _re.Match) -> str:
        name = match.group(1)
        qid = name_to_id.get(name)
        if qid is not None:
            shift = _resolve_shift_str(match.group(2))
            tokens_list.append(_in.Token(qid, shift))
            return _REPLACE_NAME.format(
                qid=qid,
                shift=shift,
                eid=_EQUATION_REF,
            )
        else:
            error_log.append(name + " in " + human)
            tokens_list.append(_in.Token(None, None))
            return _REPLACE_UKNOWN

    xtring = _QUANTITY_NAME_PATTERN.sub(_x_from_human, human)
    xtring = _postprocess_xtring(xtring)
    return xtring, set(tokens_list), error_log, tokens_list
    #]


def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("[", "",).replace("]", "")) if shift_str is not None else 0


def _postprocess_xtring(equation: str) -> str:
    #[
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    lhs_rhs = equation.split("=", maxsplit=1)
    if len(lhs_rhs)==2:
        equation = "-(" + lhs_rhs[0] + ")+" + lhs_rhs[1]
    return equation
    #]


def generate_all_eids(
    equations: Equations,
    /,
) -> Iterable[int]:
    return (eqn.id for eqn in equations)


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
    return sorted(equations, key=_op.attrgetter("id"))


def get_min_shift_from_equations(
    equations: Equations,
    /,
) -> int:
    return _in.get_min_shift(generate_all_tokens_from_equations(equations))


def get_max_shift_from_equations(
    equations: Equations,
    /,
) -> int:
    return _in.get_max_shift(generate_all_tokens_from_equations(equations))


def validate_selection_of_equations(
    allowed_equations: Equations,
    custom_equations: Equations | None,
    /,
) -> tuple[Equations, Equations]:
    """
    """
    #[
    invalid_equations = list(set(custom_equations) - set(allowed_equations)) if custom_equations is not None else []
    custom_equations = list(custom_equations) if custom_equations is not None else list(allowed_equations)
    #]
    return custom_equations, invalid_equations


def lookup_eids_by_human_starts(
    equations: Equations,
    human_starts: Iterable[str],
    /,
) -> tuple[Equations, list[str]]:
    """
    """
    #[
    human_starts = list(human_starts)
    eids = [
        next((eqn.id for eqn in equations if eqn.human.startswith(hs)), None)
        for hs in human_starts
    ]
    valid_eids = [ i for i in eids if i is not None ]
    invalid_human_starts = [ h for h, i in zip(human_starts, eids) if i is None ]
    #]
    return valid_eids, invalid_human_starts


def create_incidence_matrix(
    equations: Equations,
    quantities: _qu.Quantities,
    /,
) -> None:
    """
    """
    num_equations = len(equations)
    num_quantities = len(quantities)
    incidence_matrix = _np.zeros((num_equations, num_quantities), dtype=bool)
    qids = list(_qu.generate_all_qids(quantities))
    qid_to_column = { qid: column for column, qid in enumerate(qids) }
    for row_index, eqn in enumerate(equations):
        column_indices = tuple(
            qid_to_column[tok.qid]
            for tok in eqn.incidence if tok.qid in qids
        )
        incidence_matrix[row_index, column_indices] = True
    return incidence_matrix


