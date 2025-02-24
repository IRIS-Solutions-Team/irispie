"""
Mixin for enumeration classes that represent keywords
"""


#[

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self

#]


class KindMixin:
    """
    """
    #[

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

    def _serialize_to_portable(self, /, ) -> str:
        return self.name

    @classmethod
    def _deserialize_from_portable(klass, value: str, /, ) -> Self:
        return klass[value]

    @property
    def human(self, /, ) -> str:
        return self.name.replace("_", " ").title()

    #]

