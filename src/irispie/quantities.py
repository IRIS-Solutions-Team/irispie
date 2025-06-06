"""
Model quantities
"""


#[
from __future__ import annotations

from typing import Self, Any
from typing import Protocol
from collections.abc import Iterable, Iterator, Sequence
import enum
import collections as _co
import re as _re
from dataclasses import dataclass

from . import wrongdoings as _wrongdoings
from . import attributes as _attributes
#]


QUANTITY_OCCURRENCE_PATTERN = _re.compile(
    r"\b([a-zA-Z]\w*)\b(\[[-+\d]+\])?(?!\()"
)


_PortableType = tuple[str, str, bool, str, str, ]


class QuantityKind(enum.Flag):
    """
    Classification of model quantities
    """
    #[

    UNSPECIFIED = enum.auto()

    TRANSITION_VARIABLE = enum.auto()
    MEASUREMENT_VARIABLE = enum.auto()
    UNANTICIPATED_SHOCK = enum.auto()
    ANTICIPATED_SHOCK = enum.auto()
    MEASUREMENT_SHOCK = enum.auto()
    LHS_VARIABLE = enum.auto()
    RHS_ONLY_VARIABLE = enum.auto()
    PARAMETER = enum.auto()
    EXOGENOUS_VARIABLE = enum.auto()
    UNANTICIPATED_STD = enum.auto()
    ANTICIPATED_STD = enum.auto()
    MEASUREMENT_STD = enum.auto()

    ENDOGENOUS_VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE
    ANY_VARIABLE = ENDOGENOUS_VARIABLE | EXOGENOUS_VARIABLE
    ANY_SHOCK = UNANTICIPATED_SHOCK | ANTICIPATED_SHOCK | MEASUREMENT_SHOCK
    ANY_STD = UNANTICIPATED_STD | ANTICIPATED_STD | MEASUREMENT_STD
    PARAMETER_OR_STD = PARAMETER | ANY_STD
    STOCHASTIC_SHOCK = UNANTICIPATED_SHOCK | MEASUREMENT_SHOCK
    LOGGABLE_VARIABLE = TRANSITION_VARIABLE | MEASUREMENT_VARIABLE | EXOGENOUS_VARIABLE
    LOGGABLE_VARIABLE_OR_ANY_SHOCK = LOGGABLE_VARIABLE | ANY_SHOCK

    @classmethod
    def from_keyword(
        klass,
        keyword: str,
        /,
    ) -> Self:
        """
        """
        return klass[
            keyword
            .replace("-", "_")
            .replace(" ", "_")
            .replace("!", "")
            .strip()
            .upper()
            .removesuffix("S")
        ]

    def to_keyword(self, /, ) -> str:
        return "!" + self.name.lower() + "s"

    def to_portable(self, /, ) -> str:
        return _TO_PORTABLES[self]

    @classmethod
    def from_portable(klass, portable: str, /, ) -> Self:
        return _FROM_PORTABLES[portable]

    @property
    def human(self, /, ) -> str:
        return self.name.replace("_", " ").title()

    #]


for n in QuantityKind.__members__:
    exec(f"{n} = QuantityKind.{n}")


_TO_PORTABLES = {
    QuantityKind.TRANSITION_VARIABLE: "#x",
    QuantityKind.MEASUREMENT_VARIABLE: "#y",
    QuantityKind.UNANTICIPATED_SHOCK: "#u",
    QuantityKind.ANTICIPATED_SHOCK: "#v",
    QuantityKind.MEASUREMENT_SHOCK: "#w",
    QuantityKind.PARAMETER: "#p",
    QuantityKind.EXOGENOUS_VARIABLE: "#z",
}


_FROM_PORTABLES = {
    v: k for k, v in _TO_PORTABLES.items()
}


__all__ = (
    "filter_quantities_by_name",
    "TRANSITION_VARIABLE",
    "ANTICIPATED_SHOCK",
    "ANTICIPATED_STD",
    "UNANTICIPATED_SHOCK",
    "UNANTICIPATED_STD",
    "MEASUREMENT_VARIABLE",
    "MEASUREMENT_SHOCK",
    "MEASUREMENT_STD",
    "EXOGENOUS_VARIABLE",
    "ANY_VARIABLE",
    "ANY_SHOCK",
    "STOCHASTIC_SHOCK",
    "PARAMETER",
)


@dataclass
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
    attributes: set[str] | None = None

    def wrap_logly(self, /, ) -> str:
        return wrap_logly(self.human, self.logly, )

    def copy(self, /, ) -> Self:
        """
        Shallow copy of the quantity
        """
        return type(self)(**self.__dict__, )

    def __hash__(self, /, ) -> int:
        return hash(self.__repr__)

    has_attributes = _attributes.has_attributes

    def to_portable(self, /, ) -> _PortableType:
        """
        """
        return (
            self.kind.to_portable(),
            self.human,
            self.logly,
            self.description,
            " ".join(self.attributes, ),
        )

    @classmethod
    def from_portable(klass, portable: _PortableType, /, ) -> Self:
        """
        """
        kind, human, logly, description, attributes = portable
        return klass(
            human=human,
            kind=QuantityKind.from_portable(kind, ),
            logly=logly,
            description=description,
            attributes=set(attributes.split(" ", )),
        )

    #]


def create_name_to_qid(quantities: Iterable[Quantity]) -> dict[str, int]:
    return {
        qty.human: qty.id for qty in quantities
        if qty.id is not None and qty.human is not None
    }


def create_name_to_quantity(quantities: Iterable[Quantity], /, ) -> dict[str, Quantity]:
    return { qty.human: qty for qty in quantities if qty.human is not None }


def create_qid_to_name(quantities: Iterable[Quantity]) -> dict[int, str]:
    return {
        qty.id: qty.human for qty in quantities
        if qty.id is not None and qty.human is not None
    }


def create_qid_to_description(quantities: Iterable[Quantity]) -> dict[int, str]:
    return {
        qty.id: qty.description for qty in quantities
        if qty.id is not None and qty.description is not None
    }


def create_name_to_kind(quantities: Iterable[Quantity]) -> dict[str, str]:
    return {
        qty.human: qty.kind for qty in quantities
        if qty.human is not None and qty.description is not None
    }


def create_name_to_description(quantities: Iterable[Quantity]) -> dict[str, str]:
    return {
        qty.human: qty.description for qty in quantities
        if qty.human is not None and qty.description is not None
    }


def create_qid_to_kind(quantities: Iterable[Quantity]) -> dict[int, QuantityKind]:
    return {
        qty.id: qty.kind for qty in quantities
        if qty.id is not None and qty.kind is not None
    }


def generate_quantities_of_kind(
    quantities: Iterable[Quantity],
    kind: QuantityKind | None = None,
) -> Iterable[Quantity]:
    """
    """
    return ( qty for qty in quantities if kind is None or qty.kind in kind )


def generate_names_of_kind(
    quantities: Iterable[Quantity],
    kind: QuantityKind | None = None,
) -> Iterable[str]:
    """
    """
    return ( qty.human for qty in quantities if kind is None or qty.kind in kind )


def count_quantities_of_kind(
    quantities: Iterable[Quantity],
    kind: QuantityKind | None,
    /,
) -> int:
    """
    """
    return sum(1 for _ in generate_quantities_of_kind(quantities, kind, ))


def generate_qids_by_kind(
    quantities: Iterable[Quantity],
    kind: QuantityKind | None,
) -> Iterable[int | None]:
    """
    """
    return ( qty.id for qty in generate_quantities_of_kind(quantities, kind, ) )


def generate_quantity_names_by_kind(
    quantities: Iterable[Quantity],
    kind: QuantityKind | None,
) -> Iterator[str | None]:
    """
    """
    return ( qty.human for qty in generate_quantities_of_kind(quantities, kind, ) )


def generate_all_quantity_names(quantities: Iterable[Quantity]) -> Iterator[str | None]:
    return ( qty.human for qty in quantities )


def generate_all_qids(quantities: Iterable[Quantity]) -> Iterator[int | None]:
    return ( qty.id for qty in quantities )


def get_max_qid(quantities: Iterable[Quantity]) -> int:
    return max(
        qty.id for qty in quantities
        if qty.id is not None
    )


def create_name_to_logly(quantities: Iterable[Quantity]) -> dict[int, bool]:
    return {
        qty.human: qty.logly for qty in quantities
        if qty.id is not None and qty.logly is not None
    }

def create_qid_to_logly(quantities: Iterable[Quantity]) -> dict[int, bool]:
    return {
        qty.id: qty.logly for qty in quantities
        if qty.id is not None and qty.logly is not None
    }


def generate_logly_indexes(quantities: Iterable[Quantity]) -> Iterator[int]:
    return (
        qty.id for qty in quantities
        if qty.id is not None and qty.logly
    )


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
) -> tuple[tuple[Quantity, ...], tuple[str, ...]]:
    """
    Lookup quantities by name, and return a list of quantities and a list
    of invalid names
    """
    custom_names = list(custom_names)
    name_to_quantity  = create_name_to_quantity(quantities, )
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


def generate_where_logly(
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> Iterable[int]:
    """
    """
    return (i for i, qid in enumerate(qids, ) if qid_to_logly.get(qid, False))


def wrap_logly(name, logly: bool = True, /, ) -> str:
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
    return tuple(sorted(quantities, key=lambda x: (x.kind.value, x.entry)))


def stamp_id(quantities: Iterable[Quantity], /) -> None:
    """
    """
    for i, q in enumerate(quantities, ):
        q.id = i


def to_portable(quantities: Iterable[Quantity], /, ) -> tuple[_PortableType, ...]:
    """
    """
    #[
    portable = []
    for kind in _TO_PORTABLES.keys():
        portable += [ i.to_portable() for i in quantities if i.kind == kind ]
    return tuple(portable)
    #]


def from_portable(portables: Iterable[_PortableType], /, ) -> tuple[Quantity, ...]:
    """
    """
    return tuple( Quantity.from_portable(i) for i in portables )


class AccessQuantitiesProtocol(Protocol, ):
    """
    """
    #[

    def _access_quantities(self, /, ) -> Iterable[Quantity, ...]: ...

    #]


class Mixin:
    r"""
    Mixin for objects with quantities
    """
    #[

    def _access_quantities(self, ) -> Iterable[Quantity]:
        r"""
        Default implementation of the _access_quantities method
        """
        return self._invariant.quantities

    def get_quantities(
        self: AccessQuantitiesProtocol,
        kind: QuantityKind | None = None,
    ) -> Iterable[Quantity]:
        """
        """
        quantities = self._access_quantities()
        return tuple(
            generate_quantities_of_kind(quantities, kind, )
            if kind else quantities
        )

    def get_names(
        self,
        kind: QuantityKind | None = None,
    ) -> tuple[str, ...]:
        """
        """
        return tuple(q.human for q in self.get_quantities(kind=kind, ))

    def get_log_status(
        self,
        **kwargs,
    ) -> dict[str, bool]:
        """
        """
        quantities = self._access_quantities()
        return {
            qty.human: qty.logly
            for qty in quantities
            if qty.kind in QuantityKind.LOGGABLE_VARIABLE
        }

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        """
        Create a dictionary mapping quantity names to quantity ids
        """
        quantities = self._access_quantities()
        return create_name_to_qid(quantities, )

    def create_qid_to_name(self, /, ) -> dict[int, str]:
        """
        Create a dictionary mapping quantity ids to quantity names
        """
        quantities = self._access_quantities()
        return create_qid_to_name(quantities, )

    def create_qid_to_kind(self, /, ) -> dict[int, str]:
        """
        Create a dictionary mapping quantity ids to quantity kinds
        """
        quantities = self._access_quantities()
        return create_qid_to_kind(quantities, )

    def create_qid_to_description(self, /, ) -> dict[int, str]:
        """
        Create a dictionary mapping quantity ids to quantity descriptions
        """
        quantities = self._access_quantities()
        return create_qid_to_description(quantities, )

    def create_name_to_description(self, ) -> dict[str, str]:
        """
        """
        quantities = self._access_quantities()
        return create_name_to_description(quantities, )

    def create_qid_to_logly(self, /, ) -> dict[int, bool]:
        """
        Create a dictionary mapping from quantity id to quantity log-status
        """
        quantities = self._access_quantities()
        return create_qid_to_logly(quantities, )

    def get_logly_indexes(self, /) -> tuple[int, ...]:
        """
        Create a tuple of indexes for logly quantities
        """
        quantities = self._access_quantities()
        return tuple(generate_logly_indexes(quantities, ))

    #]



