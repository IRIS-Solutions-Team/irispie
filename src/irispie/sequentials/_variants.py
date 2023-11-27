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

    def assign(self, *args, **kwargs, ) -> None:
        """
        """
        for a in args:
            self._assign_from_kwargs(**a, )
        self._assign_from_kwargs(**kwargs, )

    def _assign_from_kwargs(self, **kwargs, ) -> None:
        """
        """
        existing_keys = set(self.parameters.keys())
        custom_keys = set(kwargs.keys())
        for n in existing_keys & custom_keys:
            self.parameters[n] = kwargs[n]

    #]

