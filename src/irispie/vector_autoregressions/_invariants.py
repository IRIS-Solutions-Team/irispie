"""
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )

from ..conveniences import descriptions as _descriptions
from ..quantities import (Quantity, QuantityKind, )

if TYPE_CHECKING:
    from collections.abc import (Iterable, )
#]


class Invariant(
    _descriptions.DescriptionMixin,
):
    """
    """
    #[

    __slots__ = (
        "quantities",
        "order",
        "has_constant",
    )

    def __init__(
        self,
        endogenous_names: Iterable[str],
        *,
        exogenous_names: Iterable[str] | None = None,
        order: int = 1,
        constant: bool = True,
    ) -> None:
        """
        """
        exogenous_names = exogenous_names or ()
        self.quantities = _create_quantities(endogenous_names, exogenous_names, )
        self.order = int(order)
        self.has_constant = bool(constant)

    #]


def _create_quantities(
    endogenous_names: Iterable[str],
    exogenous_names: Iterable[str] | None = None,
) -> tuple[Quantity, ...]:
    """
    """
    endogenous_names = tuple(endogenous_names)
    exogenous_names = tuple(exogenous_names or ())
    #
    endogenous_quantities = tuple(
        Quantity(id=qid, human=name, kind=QuantityKind.TRANSITION_VARIABLE, )
        for qid, name in enumerate(endogenous_names)
    )
    num_endogenous = len(endogenous_quantities)
    exogenous_quantities = tuple(
        Quantity(id=qid, human=name, kind=QuantityKind.EXOGENOUS_VARIABLE, )
        for qid, name in enumerate(exogenous_names, start=num_endogenous, )
    )
    return endogenous_quantities + exogenous_quantities

