r"""
Vector autoregressive models
"""


#[

from __future__ import annotations

from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants

from ._invariants import Invariant
from ._variants import Variant

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from collections.abc import Iterable

#]


__all__ = (
    "VecAutoreg",
)

class VecAutoreg(
    _has_invariant.Mixin,
    _has_variants.Mixin,
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
        num_variants: int = 1,
        *args,
        **kwargs,
    ) -> Self:
        """
        """
        invariant = Invariant(*args, **kwargs, )
        variants = [
            Variant(
                order=invariant.order,
                intercept=invariant.has_intercept,
            )
            for _ in range(num_variants)
        ]
        return klass(invariant=invariant, variants=variants, )

#]

