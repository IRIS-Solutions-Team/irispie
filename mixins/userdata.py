"""
General mixins
"""

#[
from __future__ import annotations

from typing import Self, NoReturn
#]


class DescriptMixin:
    """
    """
    #[
    _descript_: str = ""

    def set_descript(self, descript: str) -> NoReturn:
        self._descript_ = str(descript)

    def get_descript(self) -> str:
        return str(self._descript_)
    #]

