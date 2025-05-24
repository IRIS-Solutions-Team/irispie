r"""
Structural vector autoregression models
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
    "StructVAR",
)


class StructVAR(
    _has_invariant.Mixin,
    _has_variants.Mixin,
):
    r"""
    """
    #[

    pass

    #]

