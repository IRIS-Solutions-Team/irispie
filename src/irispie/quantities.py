"""
Model quantities
"""


#[
from __future__ import annotations

import enum
import dataclasses as _dc

from typing import (TypeAlias, )
from collections.abc import (Iterable, )
#]


class QuantityKind(enum.Flag):
    """
    Classification of model quantities
    """
    #[
    UNSPECIFIED = enum.auto()
    TRANSITION_VARIABLE = enum.auto()
    TRANSITION_SHOCK = enum.auto()
    MEASUREMENT_VARIABLE = enum.auto()
    MEASUREMENT_SHOCK = enum.auto()
    EXOGENOUS_VARIABLE = enum.auto()
    PARAMETER = enum.auto()
    TRANSITION_STD = enum.auto()
    MEASUREMENT_STD = enum.auto()

    ENDOGENOUS_VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    VARIABLE = ENDOGENOUS_VARIABLE | EXOGENOUS_VARIABLE
    STD = TRANSITION_STD | MEASUREMENT_STD
    PARAMETER_OR_STD = PARAMETER | STD
    SHOCK = TRANSITION_SHOCK | MEASUREMENT_SHOCK
    #]


_export_kinds  = [
    "TRANSITION_VARIABLE", "TRANSITION_SHOCK", "TRANSITION_STD",
    "MEASUREMENT_VARIABLE", "MEASUREMENT_SHOCK", "MEASUREMENT_STD",
]

for n in _export_kinds:
    exec(f"{n} = QuantityKind.{n}")

__all__ = ["filter_quantities_by_name", ] + _export_kinds


@_dc.dataclass(slots=True, )
class Quantity:
    """
    """
    #[
    id: int | None = None
    human: str | None = None
    kind: QuantityKind = QuantityKind.UNSPECIFIED
    logly: bool | None = None
    description: str | None = None
    entry: int | None = None

    def set_id(self, qid: int) -> Self:
        self.id = qid
        return self

    def set_logly(self, logly: bool) -> Self:
        self.logly = logly
        return self

    def print_name_maybe_log(self, /, ) -> str:
        return print_name_maybe_log(self.human, self.logly, )

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__)
    #]


Quantities: TypeAlias = Iterable[Quantity]
Humans: TypeAlias = Iterable[str]
HumansNotFound: TypeAlias = Iterable[str]


def create_name_to_qid(quantities: Quantities) -> dict[str, int]:
    return { qty.human: qty.id for qty in quantities }


def create_name_to_quantity(quantities: Quantities) -> dict[str, int]:
    return { qty.human: qty for qty in quantities }


def create_qid_to_name(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.human for qty in quantities }


def create_qid_to_description(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.description for qty in quantities }


def create_qid_to_kind(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.kind for qty in quantities }


def generate_quantities_of_kind(quantities: Quantities, kind: QuantityKind | None) -> Quantities:
    is_of_kind = (lambda qty: qty.kind in kind) if kind is not None else lambda k: True
    return ( qty for qty in quantities if is_of_kind(qty) )


def generate_qids_by_kind(quantities: Quantities, kind: QuantityKind) -> list[int]:
    return ( qty.id for qty in quantities if qty.kind in kind )


def generate_quantity_names_by_kind(quantities: Quantities, kind: QuantityKind) -> list[str]:
    return ( qty.human for qty in quantities if qty.kind in kind )


def generate_all_quantity_names(quantities: Quantities) -> Iterable[str]:
    return ( qty.human for qty in quantities )


def generate_all_qids(quantities: Quantities) -> Iterable[int]:
    return ( qty.id for qty in quantities )


def get_max_qid(quantities: Quantities) -> int:
    return max(qty.id for qty in quantities)


def create_qid_to_logly(quantities: Quantities) -> dict[int, bool]:
    return { qty.id: qty.logly for qty in quantities }


def change_logly(
    quantities: Quantities,
    new_logly: bool,
    qids: Iterable[int],
    /
) -> Quantities:
    qids = set(qids)
    return [
        qty if qty.id not in qids or qty.logly is None else Quantity(qty.id, qty.human, qty.kind, new_logly)
        for qty in quantities
    ]


def validate_selection_of_quantities(
    allowed_quantities: Quantities,
    custom_quantities: Quantities | None,
    /,
) -> tuple[Quantities, Quantities]:
    """
    """
    invalid_quantities = list(set(custom_quantities) - set(allowed_quantities)) if custom_quantities is not None else []
    custom_quantities = list(custom_quantities) if custom_quantities is not None else list(allowed_quantities)
    return custom_quantities, invalid_quantities


def lookup_quantities_by_name(
    quantities: Quantities,
    custom_names: Iterable[str],
    /,
) -> tuple[Quantities, tuple[str]]:
    """
    Lookup quantities by name, and return a list of quantities and a list
    of invalid names
    """
    custom_names = list(custom_names)
    name_to_quantity  = create_name_to_quantity(quantities)
    custom_quantities = tuple(
        name_to_quantity[n]
        for n in custom_names if n in name_to_quantity
    )
    invalid_names = tuple(
        n for n in custom_names if n not in name_to_quantity
    )
    return custom_quantities, invalid_names


def lookup_qids_by_name(
    quantities: Quantities,
    custom_names: Iterable[str],
    /,
) -> tuple[tuple[int], tuple[str]]:
    """
    Lookup quantities by name, and return a list of quantities and a list
    of invalid names
    """
    custom_names = list(custom_names)
    name_to_qid  = create_name_to_qid(quantities)
    custom_qids = tuple(
        name_to_qid[n]
        for n in custom_names if n in name_to_qid
    )
    invalid_names = tuple(
        n for n in custom_names if n not in name_to_qid
    )
    return custom_qids, invalid_names


def filter_quantities_by_name(
    quantities: Quantities,
    /,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] | None = None,
) -> Quantities:
    """
    """
    include_names = set(include_names) if include_names is not None else None
    exclude_names = set(exclude_names) if exclude_names is not None else None
    inclusion_test = lambda name: include_names is None or name in include_names
    exclusion_test = lambda name: exclude_names is None or name not in exclude_names
    return [ qty for qty in quantities if inclusion_test(qty.human) and exclusion_test(qty.human) ]


def generate_index_logly(
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> Iterable[int]:
    """
    """
    return (i for i, qid in enumerate(qids, ) if qid_to_logly[qid])


def print_name_maybe_log(name, logly, /, ) -> str:
    return f"log({name})" if logly else name


