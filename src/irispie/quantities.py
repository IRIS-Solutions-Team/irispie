"""
Model quantities
"""


#[
from __future__ import annotations

from typing import (TypeAlias, )
from collections.abc import (Iterable, )
import enum
import collections as _co
import dataclasses as _dc

from . import wrongdoings as _wrongdoings
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
    LHS_VARIABLE = enum.auto()
    RHS_ONLY_VARIABLE = enum.auto()
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

    def set_logly(self, logly: bool) -> Self:
        self.logly = logly
        return self

    def print_name_maybe_log(self, /, ) -> str:
        return print_name_maybe_log(self.human, self.logly, )

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__)
    #]


def create_name_to_qid(quantities: Iterable[Quantity]) -> dict[str, int]:
    return { qty.human: qty.id for qty in quantities }


def create_name_to_quantity(quantities: Iterable[Quantity]) -> dict[str, int]:
    return { qty.human: qty for qty in quantities }


def create_qid_to_name(quantities: Iterable[Quantity]) -> dict[int, str]:
    return { qty.id: qty.human for qty in quantities }


def create_qid_to_description(quantities: Iterable[Quantity]) -> dict[int, str]:
    return { qty.id: qty.description for qty in quantities }


def create_qid_to_kind(quantities: Iterable[Quantity]) -> dict[int, str]:
    return { qty.id: qty.kind for qty in quantities }


def generate_quantities_of_kind(quantities: Iterable[Quantity], kind: QuantityKind | None) -> Iterable[Quantity]:
    if kind is not None:
        def is_of_kind(qty: Quantity, /, ) -> bool: return qty.kind in kind
    else:
        def is_of_kind(qty: Quantity, /, ) -> bool: return True
    return ( qty for qty in quantities if is_of_kind(qty) )


def generate_qids_by_kind(quantities: Iterable[Quantity], kind: QuantityKind) -> list[int]:
    return ( qty.id for qty in quantities if qty.kind in kind )


def generate_quantity_names_by_kind(quantities: Iterable[Quantity], kind: QuantityKind) -> list[str]:
    return ( qty.human for qty in quantities if qty.kind in kind )


def generate_all_quantity_names(quantities: Iterable[Quantity]) -> Iterable[str]:
    return ( qty.human for qty in quantities )


def generate_all_qids(quantities: Iterable[Quantity]) -> Iterable[int]:
    return ( qty.id for qty in quantities )


def get_max_qid(quantities: Iterable[Quantity]) -> int:
    return max(qty.id for qty in quantities)


def create_qid_to_logly(quantities: Iterable[Quantity]) -> dict[int, bool]:
    return { qty.id: qty.logly for qty in quantities }


def change_logly(
    quantities: Iterable[Quantity],
    new_logly: bool,
    qids: Iterable[int],
    /
) -> Iterable[Quantity]:
    qids = set(qids)
    return [
        qty if qty.id not in qids or qty.logly is None else Quantity(qty.id, qty.human, qty.kind, new_logly)
        for qty in quantities
    ]


def validate_selection_of_quantities(
    allowed_quantities: Iterable[Quantity],
    custom_quantities: Iterable[Quantity] | None,
    /,
) -> tuple[Iterable[Quantity], Iterable[Quantity]]:
    """
    """
    invalid_quantities = list(set(custom_quantities) - set(allowed_quantities)) if custom_quantities is not None else []
    custom_quantities = list(custom_quantities) if custom_quantities is not None else list(allowed_quantities)
    return custom_quantities, invalid_quantities


def lookup_quantities_by_name(
    quantities: Iterable[Quantity],
    custom_names: Iterable[str],
    /,
) -> tuple[Iterable[Quantity], tuple[str]]:
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
    quantities: Iterable[Quantity],
    custom_names: Iterable[str],
    /,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """
    Lookup quantities by name, and return a list of quantities and a list
    of invalid names
    """
    custom_names = tuple(custom_names)
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
    quantities: Iterable[Quantity],
    /,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] | None = None,
) -> Iterable[Quantity]:
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


def check_unique_names(quantities: Iterable[Quantity], /, ) -> None:
    """
    """
    #[
    name_counter = _co.Counter(q.human for q in quantities)
    if any(c>1 for c in name_counter.values()):
        duplicates = [ n for n, c in name_counter.items() if c>1 ]
        raise _wrongdoings.IrisPieError(
            ["These names are declared multiple times"] + duplicates
        )
    #]


def reorder_by_kind(quantities: Iterable[Quantity], /) -> Iterable[Quantity]:
    return list(sorted(quantities, key=lambda x: (x.kind.value, x.entry)))


def stamp_id(quantities: Iterable[Quantity], /) -> None:
    """
    """
    for i, q in enumerate(quantities, ):
        q.id = i

