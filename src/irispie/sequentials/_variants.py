"""
"""


#[
from __future__ import annotations

from typing import (Any, )
from collections.abc import (Iterable, )
import copy as _co

from .. import quantities as _quantities
from ..databoxes import main as _databoxes
#]


class Variant:
    """
    """
    #[

    __slots__ = (
        "parameters",
    )

    def __init__(
        self,
        parameter_names: Iterable[str],
        /,
    ) -> None:
        """
        """
        self.parameters = { n: None for n in parameter_names }

    def copy(self, /, ) -> Self:
        """
        """
        return _co.deepcopy(self, )

    def assign_from_dict_like(self, dict_like: dict[str, Any], ) -> None:
        """
        """
        existing_keys = set(self.parameters.keys())
        custom_keys = set(dict_like.keys())
        for n in (custom_keys & existing_keys):
            self.parameters[n] = dict_like[n]

    #]

