"""
"""


#[
from __future__ import annotations

from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants

from ._invariants import Invariant
from ._variants import Variant
#]


__all__ = (
    "VecAutoreg",
)

class VecAutoreg(
    _has_invariant.HasInvariantMixin,
    _has_variants.HasVariantsMixin,
):
    """
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(
        self,
        invariant: Invariant,
        variants: list[Variant] | None = None,
    ) -> None:
        """
        """
        self._invariant = invariant
        self._variants = variants or []

    @classmethod
    def from_names(
        klass,
        endogenous_names: Iterable[str],
        num_variants: int = 1,
        **kwargs,
    ) -> VecAutoreg:
        """
        """
        invariant = Invariant(
            endogenous_names=endogenous_names,
            **kwargs,
        )
        variants = [
            Variant(
                order=invariant.order,
                constant=invariant.has_constant,
            )
            for _ in range(num_variants)
        ]
        return klass(invariant=invariant, variants=variants, )

#]

