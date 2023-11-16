"""
Description mixin
"""


#[
from __future__ import annotations
#]


class DescriptionMixin:
    """
    """
    #[

    def get_description(self, /, ) -> str:
        return str(self._description or "")

    def set_description(self, description: str, /, ) -> None:
        self._description = str(description or "")

    #]

