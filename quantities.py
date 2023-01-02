
from enum import (
    Flag as en_Flag,
    auto as en_auto,
)


from dataclasses import (
    dataclass as dc_dataclass,
)


from typing import (
    Iterable as tp_Iterable,
)



class QuantityKind(en_Flag):
    UNSPECIFIED = en_auto()
    TRANSITION_VARIABLE = en_auto()
    TRANSITION_SHOCK = en_auto()
    MEASUREMENT_VARIABLE = en_auto()
    MEASUREMENT_SHOCK = en_auto()
    PARAMETER = en_auto()

    VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    SHOCK = TRANSITION_SHOCK | MEASUREMENT_SHOCK
    FIRST_ORDER_SYSTEM = VARIABLE | SHOCK


@dc_dataclass
class Quantity():
    """
    """
    id: int
    human: str
    kind: QuantityKind = QuantityKind.UNSPECIFIED
    log_flag: bool = False


def create_name_to_id(quantities: tp_Iterable[Quantity]) -> dict[str, int]:
    return { qty.human: qty.id for qty in quantities }


def create_id_to_name(quantities: tp_Iterable[Quantity]) -> dict[int, str]:
    return { qty.id: qty.human for qty in quantities }


def create_id_to_kind(quantities: tp_Iterable[Quantity]) -> dict[int, str]:
    return { qty.id: qty.kind for qty in quantities }


def generate_quantity_ids_by_kind(quantities: tp_Iterable[Quantity], kind: QuantityKind) -> list[int]:
    return ( qty.id for qty in quantities if qty.kind in kind )


def generate_quantity_names_by_kind(quantities: tp_Iterable[Quantity], kind: QuantityKind) -> list[str]:
    return ( qty.human for qty in quantities if qty.kind in kind )


def generate_all_quantity_names(quantities: tp_Iterable[Quantity]) -> tp_Iterable[int]:
    return ( qty.human for qty in quantities )


def generate_all_quantity_ids(quantities: tp_Iterable[Quantity]) -> tp_Iterable[int]:
    return ( qty.id for qty in quantities )


