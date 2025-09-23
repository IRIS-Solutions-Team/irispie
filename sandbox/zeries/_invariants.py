"""
Time series variants
"""


#[
from __future__ import annotations

from typing import (Self, Any, )
import copy as _co

from ..conveniences import descriptions as _descriptions
#]


class Invariant(
    _descriptions.DescriptionMixin,
):
    """
    """
    #[

    __slots__ = (
        "_description",
        "_custom_data",
    )

    def __init__(
        self: Self,
        /,
        *,
        description: str | None = None,
        custom_data: dict[str, Any] | None = None,
    ) -> None:
        """
        """
        self._description = description or ""
        self._custom_data = custom_data or {}

    def copy(self, /, ) -> Self:
        """
        """
        return _co.deepcopy(self, )

    #]

