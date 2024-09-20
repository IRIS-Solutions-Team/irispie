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
        return str(self.__description__ or "")

    def set_description(self, description: str, /, ) -> None:
        self.__description__ = str(description or "")

    #]

