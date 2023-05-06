"""
General mixins
"""

#[
from __future__ import annotations

from typing import Self, NoReturn
#]


class DescriptorMixin:
    """
    """
    #[
    _descriptor_: str = ""

    def set_descriptor(self, descriptor: str) -> NoReturn:
        self._descriptor_ = str(descriptor)

    def get_descriptor(self) -> str:
        return str(self._descriptor_)
    #]

