"""
First-order stacked-time system
"""


#[

from __future__ import annotations

from .. import has_variants as _has_variants
from ..simultaneous.main import Simultaneous
from .. import quantities as _quantities
from ..quantities import QuantityKind, Quantity
from ..databoxes.main import Databox
from ..dataslates.main import Dataslate
from .. import dates as _dates
from ..dates import Period

from ._invariants import Invariant
from ._variants import Variant
from ._slatable_protocols import Slatable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..simultaneous.main import Simultaneous

#]


__all__ = [
    "Stacker",
]


class Stacker(
    _has_variants.Mixin,
    _quantities.Mixin,
):
    """
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    @classmethod
    def from_simultaneous(
        klass,
        model: Simultaneous,
        num_periods: int,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = Invariant.from_simultaneous(model, num_periods, **kwargs, )
        self._variants = [
            Variant.from_solution(self._invariant, solution, )
            for solution in model.get_solution(unpack_singleton=False, )
        ]
        return self

    def _access_quantities(self, /, ) -> list[Quantity]:
        """
        Implement quantities.AccessQuantitiesProtocol
        """
        return self._invariant.quantities

    @property
    def max_lag(self, /, ) -> int:
        return self._invariant.max_lag

    @property
    def max_lead(self, /, ) -> int:
        return self._invariant.max_lead

    def get_base_periods(self, start: Period, /, ) -> list[Period]:
        return self._invariant.get_base_periods(start, )

    def create_dataslate(
        self,
        databox: Databox,
        start: Period,
        num_variants: int | None = None,
        shocks_from_data: bool = False,
        # stds_from_data: bool = False,
        # parameters_from_data: bool = False,
    ) -> Dataslate:
        """
        """
        num_variants = self.resolve_num_variants_in_context(num_variants, )
        base_periods = self.get_base_periods(start, )
        #
        slatable = Slatable(
            self,
            shocks_from_data=shocks_from_data,
        )
        #
        dataslate = Dataslate.from_databox_for_slatable(
            slatable, databox, base_periods,
            num_variants=num_variants,
        )
        #
        return dataslate

    #]

