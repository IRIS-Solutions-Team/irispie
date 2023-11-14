"""
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
from ..databoxes import main as _databoxes
#]


class AssignMixin:
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
        keys = self._variants[0].parameters.keys()
        return _databoxes.Databox(**{
            k: [ v.parameters[k] for v in self._variants ]
            for k in keys
        })

    #]

