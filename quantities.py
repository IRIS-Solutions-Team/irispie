"""
Model quantities
"""


#[
from __future__ import annotations

import enum
import dataclasses

from typing import TypeAlias
from collections.abc import Iterable
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

    LOGLY_VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE | EXOGENOUS_VARIABLE
    SHOCK = TRANSITION_SHOCK | MEASUREMENT_SHOCK
    STD = TRANSITION_STD | MEASUREMENT_STD
    PARAMETER_OR_STD = PARAMETER | STD

    IN_TRANSITION_EQUATIONS = TRANSITION_VARIABLE | TRANSITION_SHOCK | PARAMETER | TRANSITION_STD
    IN_MEASUREMENT_EQUATIONS = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE | MEASUREMENT_SHOCK | PARAMETER | MEASUREMENT_STD

    TRANSITION_SYSTEM_QUANTITY = TRANSITION_VARIABLE | TRANSITION_SHOCK
    MEASUREMENT_SYSTEM_QUANTITY = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE | MEASUREMENT_SHOCK
    SYSTEM_QUANTITY = TRANSITION_SYSTEM_QUANTITY | MEASUREMENT_SYSTEM_QUANTITY

    STEADY_EVALUATOR = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    #]


@dataclasses.dataclass
class Quantity:
    """
    """
    #[
    id: int | None = None
    human: str | None = None
    kind: QuantityKind = QuantityKind.UNSPECIFIED
    logly: bool | None = None
    descript: str | None = None
    entry: int | None = None

    def set_id(self, qid: int) -> Self:
        self.id = qid
        return self

    def set_logly(self, logly: bool) -> Self:
        self.logly = logly
        return self
    #]


Quantities: TypeAlias = Iterable[Quantity]


def create_name_to_qid(quantities: Quantities) -> dict[str, int]:
    return { qty.human: qty.id for qty in quantities }


def create_qid_to_name(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.human for qty in quantities }


def create_qid_to_descript(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.descript for qty in quantities }


def create_qid_to_kind(quantities: Quantities) -> dict[int, str]:
    return { qty.id: qty.kind for qty in quantities }


def generate_qids_by_kind(quantities: Quantities, kind: QuantityKind) -> list[int]:
    return ( qty.id for qty in quantities if qty.kind in kind )


def generate_quantity_names_by_kind(quantities: Quantities, kind: QuantityKind) -> list[str]:
    return ( qty.human for qty in quantities if qty.kind in kind )


def generate_all_quantity_names(quantities: Quantities) -> Iterable[int]:
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

