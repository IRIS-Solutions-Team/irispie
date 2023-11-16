"""
Model equations
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
from collections.abc import (Iterable, )
import enum as _en
import re as _re
import numpy as _np
import dataclasses as _dc
import itertools as _it
import operator as _op
import copy as _cp

from .incidences import main as _incidence
from . import quantities as _quantities
from . import attributes as _attributes
from . import wrongdoings as _wrongdoings
#]


EVALUATOR_PREAMBLE = "lambda x, t, L: "
_EVALUATOR_FORMAT = EVALUATOR_PREAMBLE + "[{joined_xtrings}]"
_REPLACE_UKNOWN = "?"


class EquationKind(_en.Flag):
    """
    Classification of model equations
    """
    #[

    TRANSITION_EQUATION = _en.auto()
    MEASUREMENT_EQUATION = _en.auto()

    ENDOGENOUS_EQUATION = TRANSITION_EQUATION | MEASUREMENT_EQUATION

    @property
    def human(self, /, ) -> str:
        return self.name.replace("_", " ").title()

    #]


__all__  = [
    "TRANSITION_EQUATION", "MEASUREMENT_EQUATION",
]
for n in __all__:
    exec(f"{n} = EquationKind.{n}")


@_dc.dataclass(slots=True, )
class Equation(
    _attributes.AttributesMixin,
):
    """
    """
    #[

    id: int | None = None
    human: str | None = None
    kind: EquationKind | None = None
    description: str | None = None
    xtring: str | None = None
    incidence: _incidence.Tokens | None = None
    entry: int | None = None
    attributes: set[str] = ()

    def finalize(self, name_to_id: dict[str, int], ) -> None:
        """
        Finalize the equation by creating an xtring from the human string, and and composing the incidence matrix
        Any undeclared names are assumed to be caught before this step
        """
        self.xtring, self.incidence, *_ = xtring_from_human(self.human, name_to_id, )

    def create_equator_func(self, /, *, context: dict[str, Callable]) -> Callable:
        equator_func_string = create_equator_func_string([self.xtring], context)
        eval(equator_func_string, context, )

    def copy(self, /, ) -> Self:
        """
        Shallow copy of the equation
        """
        return _cp.copy(self, )

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__)

    #]


def generate_all_tokens_from_equations(equations: Iterable[Equation], /, ) -> _incidence.Tokens:
    return _it.chain.from_iterable(eqn.incidence for eqn in equations)


def finalize_dynamic_equations(
    equations: Iterable[Equation],
    name_to_id: dict[str, int],
    /,
) -> None:
    _finalize_equations_from_humans(equations, name_to_id, )
    _replace_steady_ref(equations, )


def finalize_steady_equations(
    equations: Iterable[Equation],
    name_to_id: dict[str, int],
    /,
) -> None:
    _finalize_equations_from_humans(equations, name_to_id, )
    _remove_steady_ref(equations, )


def _replace_steady_ref(equations: Iterable[Equation]) -> None:
    STEADY_REF_PATTERN = _re.compile(r"&x\[([^,\]]+),[^\]]+\]")
    for eqn in equations:
        eqn.xtring = _re.sub(STEADY_REF_PATTERN, lambda match: "L["+match.group(1)+"]", eqn.xtring)


def _remove_steady_ref(equations: Iterable[Equation]) -> None:
    STEADY_REF_PATTERN = _re.compile(r"&x\b")
    for eqn in equations:
        eqn.xtring = _re.sub(STEADY_REF_PATTERN, "x", eqn.xtring)


def _finalize_equations_from_humans(
    equations: Iterable[Equation],
    name_to_id: dict[str, int],
) -> None:
    """
    """
    #[
    for eqn in equations:
        eqn.finalize(name_to_id, )
    #]


def generate_names_from_human(human: str) -> Iterable[str]:
    """
    Generate all names from a single human string
    """
    return (f[0] for f in _quantities.QUANTITY_OCCURRENCE_PATTERN.findall(human))


def generate_all_names_from_equations(equations: Iterable[Equation]) -> list[str]:
    """
    Extract all names from a list of equations
    """
    return generate_names_from_human(" ".join(e.human for e in equations))


def create_name_to_qid_from_equations(equations: Iterable[Equation]) -> dict[str, int]:
    """
    """
    all_names = sorted(list(set(generate_all_names_from_equations(equations))))
    return { name: qid for qid, name in enumerate(all_names) }


def create_equator_func_string(xtrings: str) -> str:
    """
    """
    return _EVALUATOR_FORMAT.format(joined_xtrings=" , ".join(xtrings))


def create_eid_to_wrt_tokens(
    equations: Iterable[Equation],
    all_wrt_tokens: _incidence.Tokens,
) -> dict[int, _incidence.Tokens]:
    """
    """
    #[
    eid_to_wrt_tokens = {}
    for eqn in equations:
        eid_to_wrt_tokens[eqn.id] = _incidence.sort_tokens(
            wrt for wrt in all_wrt_tokens
            if wrt in eqn.incidence
        )
    return eid_to_wrt_tokens
    #]


def create_human_to_eid(
    equations: Iterable[Equation],
    /,
) -> dict[str, int]:
    return { eqn.human: eqn.id for eqn in equations }


def xtring_from_human(
    human: str,
    name_to_id: dict[str, int],
) -> tuple[str, set[_incidence.Token]]:
    """
    Convert human string to xtring and retrieve incidence tokens
    """
    #[
    tokens_list: list[_incidence.Token] = []

    def _replace_human_with_x(match: _re.Match) -> str:
        name = match.group(1)
        qid = name_to_id[name]
        shift = _resolve_shift_str(match.group(2))
        new_token = _incidence.Token(qid, shift)
        tokens_list.append(new_token)
        return new_token.print_xtring()

    xtring = _quantities.QUANTITY_OCCURRENCE_PATTERN.sub(_replace_human_with_x, human)
    xtring = _postprocess_xtring(xtring)
    return xtring, set(tokens_list), tokens_list
    #]


def _resolve_shift_str(shift_str: str) -> int:
    return int(shift_str.replace("[", "",).replace("]", "")) if shift_str is not None else 0


def _postprocess_xtring(equation: str) -> str:
    #[
    equation = equation.replace("^", "**")
    equation = equation.replace(" ", "")
    equation = equation.replace("===", "=")
    lhs_rhs = equation.split("=", maxsplit=1, )
    if len(lhs_rhs)==2:
        equation = "-(" + lhs_rhs[0] + ")+" + lhs_rhs[1]
    return equation
    #]


def generate_all_eids(
    equations: Iterable[Equation],
    /,
) -> Iterable[int]:
    return (eqn.id for eqn in equations)


def generate_eids_by_kind(
    equations: Iterable[Equation],
    kind: EquationKind,
) -> Iterable[int]:
    return (eqn.id for eqn in equations if eqn.kind in kind)


def generate_equations_of_kind(
    equations: Iterable[Equation],
    kind: EquationKind | None,
    /,
) -> Iterable[Equation]:
    return (eqn for eqn in equations if kind is None or eqn.kind in kind)


def count_equations_of_kind(
    equations: Iterable[Equation],
    kind: EquationKind,
    /,
) -> int:
    return sum(1 for eqn in generate_equations_of_kind(equations, kind, ))


def sort_equations(
    equations: Iterable[Equation],
    /,
) -> Iterable[Equation]:
    return sorted(equations, key=_op.attrgetter("id"))


def get_min_shift_from_equations(
    equations: Iterable[Equation],
    /,
) -> int:
    return _incidence.get_min_shift(generate_all_tokens_from_equations(equations))


def get_max_shift_from_equations(
    equations: Iterable[Equation],
    /,
) -> int:
    return _incidence.get_max_shift(generate_all_tokens_from_equations(equations))


def validate_selection_of_equations(
    allowed_equations: Iterable[Equation],
    custom_equations: Iterable[Equation] | None,
    /,
) -> tuple[Iterable[Equation], Iterable[Equation]]:
    """
    """
    #[
    invalid_equations = list(set(custom_equations) - set(allowed_equations)) if custom_equations is not None else []
    custom_equations = list(custom_equations) if custom_equations is not None else list(allowed_equations)
    #]
    return custom_equations, invalid_equations


def lookup_eids_by_human_starts(
    equations: Iterable[Equation],
    human_starts: Iterable[str],
    /,
) -> tuple[Iterable[Equation], list[str]]:
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
    equations: Iterable[_equations.Equation],
    quantities: Iterable[_quantities.Quantity],
    /,
    shift_test: Callable | None = None,
    data_type: type = bool,
) -> _np.ndarray:
    """
    """
    #[
    num_equations = len(equations)
    num_quantities = len(quantities)
    incidence_matrix = _np.zeros((num_equations, num_quantities), dtype=data_type, )
    qids = tuple(_quantities.generate_all_qids(quantities))
    qid_to_column = { qid: column for column, qid in enumerate(qids) }
    for row_index, eqn in enumerate(equations):
        column_indices = tuple(
            qid_to_column[tok.qid]
            for tok in eqn.incidence
            if tok.qid in qids and (shift_test(tok) if shift_test else True)
        )
        incidence_matrix[row_index, column_indices] = True
    return incidence_matrix
    #]


def reorder_by_kind(equations: Iterable[Equation], /) -> Iterable[Equation]:
    return list(sorted(equations, key=lambda x: (x.kind.value, x.entry)))


def stamp_id(equations: Iterable[Equation], /) -> None:
    """
    """
    for i, e in enumerate(equations, ):
        e.id = i


def create_human_to_description(equations: Iterable[Equation]) -> dict[str, str]:
    return { eqn.human: eqn.description for eqn in equations }

