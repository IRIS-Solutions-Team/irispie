"""
Model equations
"""


#[

from __future__ import annotations

from typing import Self, Callable
from collections.abc import Iterable, Sequence
import enum as _en
import re as _re
import numpy as _np
import itertools as _it
import operator as _op
from dataclasses import dataclass

from .incidences.main import Token
from .incidences import main as _incidences
from . import quantities as _quantities
from . import attributes as _attributes
from . import wrongdoings as _wrongdoings

#]


EVALUATOR_PREAMBLE = "lambda x, t: "
_EVALUATOR_FORMAT = EVALUATOR_PREAMBLE + "[{joined_xtrings}]"
_REPLACE_UKNOWN = "?"


_PortableType = tuple[str, str | None, str, str, str]


class EquationKind(_en.Flag):
    """
    Classification of model equations
    """
    #[

    TRANSITION_EQUATION = _en.auto()
    MEASUREMENT_EQUATION = _en.auto()
    STEADY_AUTOVALUES = _en.auto()

    ENDOGENOUS_EQUATION = TRANSITION_EQUATION | MEASUREMENT_EQUATION

    @classmethod
    def from_portable(klass, portable: str, /, ) -> Self:
        return _FROM_PORTABLES[portable]

    def to_keyword(self, /, ) -> str:
        return "!" + self.name.lower() + "s"

    def to_portable(self, /, ) -> str:
        return _TO_PORTABLES[self]

    @property
    def human(self, /, ) -> str:
        return self.name.replace("_", " ").title()

    #]


_TO_PORTABLES = {
    EquationKind.TRANSITION_EQUATION: "#T",
    EquationKind.MEASUREMENT_EQUATION: "#M",
    EquationKind.STEADY_AUTOVALUES: "#A",
}


_FROM_PORTABLES = {
    v: k for k, v in _TO_PORTABLES.items()
}


__all__  = (
    "TRANSITION_EQUATION",
    "MEASUREMENT_EQUATION",
    "STEADY_AUTOVALUES",
    "ENDOGENOUS_EQUATION",
)
for n in __all__:
    exec(f"{n} = EquationKind.{n}")


@dataclass
class Equation:
    """
    """
    #[

    id: int | None = None
    human: str | None = None
    kind: EquationKind | None = None
    description: str | None = None
    xtring: str | None = None
    incidence: Iterable[Token] | None = None
    entry: int | None = None
    attributes: set[str] | None = None

    def finalize(self, name_to_id: dict[str, int], ) -> None:
        """
        Finalize the equation by creating an xtring from the human string, and and composing the incidence matrix
        Any undeclared names are assumed to be caught before this step
        """
        self.xtring, self.incidence, *_ = xtring_from_human(self.human, name_to_id, )

    def create_equator_func(self, /, *, context: dict[str, Callable]) -> Callable:
        context = (dict(context) if context else {}) | {"__builtins__": {}}
        equator_func_string = create_equator_func_string([self.xtring], context, )
        eval(equator_func_string, context, )

    def copy(self, /, ) -> Self:
        """
        Shallow copy of the equation
        """
        return type(self)(**self.__dict__, )

    def to_portable(self, complement: Self, /, ) -> _PortableType:
        """
        """
        return (
            self.kind.to_portable(),
            self.human,
            complement.human if complement.human != self.human else None,
            self.description,
            " ".join(self.attributes, ),
        )

    @classmethod
    def from_portable(klass, portable: _PortableType, /, ) -> Self:
        kind, human, complement_human, description, attributes = portable
        self = klass(
            human=human,
            kind=EquationKind.from_portable(kind),
            description=description,
            attributes=set(attributes),
        )
        complement = klass(
            human=complement_human or human,
            kind=EquationKind.from_portable(kind),
            description=description,
            attributes=set(attributes),
        )
        return self, complement,

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__, )

    has_attributes = _attributes.has_attributes

    #]


def generate_all_tokens_from_equations(equations: Iterable[Equation], /, ) -> Iterable[Token]:
    return _it.chain.from_iterable(eqn.incidence for eqn in equations)


def finalize_equations(
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
    all_wrt_tokens: Iterable[Token],
) -> dict[int, Iterable[Token], ]:
    """
    """
    #[
    eid_to_wrt_tokens = {}
    for eqn in equations:
        eid_to_wrt_tokens[eqn.id] = _incidences.sort_tokens(
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
) -> tuple[str, set[Token]]:
    """
    Convert human string to xtring and retrieve incidence tokens
    """
    #[
    tokens_list: list[Token] = []

    def _replace_human_with_x(match: _re.Match) -> str:
        name = match.group(1)
        qid = name_to_id[name]
        shift = _resolve_shift_str(match.group(2))
        new_token = Token(qid, shift)
        tokens_list.append(new_token)
        return new_token.print_xtring()

    xtring = human
    xtring = xtring.replace(":=", "=")
    xtring = _quantities.QUANTITY_OCCURRENCE_PATTERN.sub(_replace_human_with_x, xtring, )
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


def generate_all_eids(equations: Iterable[Equation], ) -> Iterable[int]:
    return (eqn.id for eqn in equations)


def generate_eids_by_kind(
    equations: Iterable[Equation],
    kind: EquationKind,
) -> Iterable[int]:
    return (eqn.id for eqn in equations if eqn.kind in kind)


def generate_equations_of_kind(
    equations: Iterable[Equation],
    kind: EquationKind | None,
) -> Iterable[Equation]:
    return (eqn for eqn in equations if kind is None or eqn.kind in kind)


def count_equations_of_kind(
    equations: Iterable[Equation],
    kind: EquationKind,
) -> int:
    return sum(1 for eqn in generate_equations_of_kind(equations, kind, ))


def sort_equations(equations: Iterable[Equation], ) -> Iterable[Equation]:
    return sorted(equations, key=_op.attrgetter("id"))


def get_min_shift_from_equations(equations: Iterable[Equation], ) -> int:
    return _incidences.get_min_shift(generate_all_tokens_from_equations(equations))


def get_max_shift_from_equations(
    equations: Iterable[Equation],
    /,
) -> int:
    return _incidences.get_max_shift(generate_all_tokens_from_equations(equations))


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


def calculate_incidence_matrix(
    equations: Iterable[Equation],
    num_quantities: int,
    token_within_quantities: Callable,
    /,
    data_type: type = bool,
) -> _np.ndarray:
    """
    """
    #[
    equations = tuple(equations)
    num_equations = len(equations)
    incidence_matrix = _np.zeros((num_equations, num_quantities), dtype=data_type, )
    for row_index, eqn in enumerate(equations):
        #
        # For each incidenc token in this equation, the
        # `token_within_quantities` function returns either an integer meaning
        # the incidence token is to be included in the incidence matrix in the
        # corresponding column, or None meaning the incidence token is not to be
        # in the incidence matrix. The Nones need to be filtered out.
        column_indices = tuple(
            token_within_quantities(inc_tok, )
            for inc_tok in eqn.incidence
        )
        column_indices = tuple(i for i in column_indices if i is not None)
        incidence_matrix[row_index, column_indices] = True
        #
    return incidence_matrix
    #]


def reorder_by_kind(equations: Iterable[Equation], /) -> Iterable[Equation]:
    return list(sorted(equations, key=lambda x: (x.kind.value, x.entry)))


def stamp_id(equations: Iterable[Equation], /) -> None:
    """
    """
    for i, e in enumerate(equations, ):
        e.id = i


def to_portable(
    dynamic_equations: Iterable[Equation],
    steady_equations: Iterable[Equation],
) -> tuple[_PortableType]:
    """
    """
    #[
    portable = []
    for kind in _TO_PORTABLES.keys():
        portable += [
            d.to_portable(s, )
            for d, s in zip(dynamic_equations, steady_equations, )
            if d.kind == kind
        ]
    return tuple(portable)
    #]


def from_portable(portable: tuple[_PortableType], /) -> tuple[Equation]:
    """
    """
    dynamic_steady = ( Equation.from_portable(p, ) for p in portable )
    return tuple(zip(*dynamic_steady))


def create_human_to_description(equations: Iterable[Equation]) -> dict[str, str]:
    return { eqn.human: eqn.description for eqn in equations }

