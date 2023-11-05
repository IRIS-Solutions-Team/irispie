"""
"""


#[
from __future__ import annotations

from typing import (Any, )
from collections.abc import (Iterable, )

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
        quantities: Iterable[_quantities.Quantity] | None = None,
        /,
    ) -> None:
        """
        """
        self.parameters = {
            qty.human: None
            for qty in quantities
            if qty.kind is _quantities.QuantityKind.PARAMETER
        }

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
        new_keys = set(kwargs.keys())
        for n in existing_keys & new_keys:
            self.parameters[n] = kwargs[n]

    #]


class VariantMixin:
    """
    """
    #[

    def assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        """
        for v in self._variants:
            v.assign(*args, **kwargs, )

    def get_parameters(self, /, ) -> _databoxes.Databox:
        """
        """
        parameters = self._variants[0].parameters
        return _databoxes.Databox.from_dict(parameters, )

    #]

