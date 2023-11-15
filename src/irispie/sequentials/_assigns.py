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
        parameter_names = self._invariant.parameter_names
        if self.is_singleton:
            return _databoxes.Databox(**{
                n: self._variants[0].parameters[n]
                for n in parameter_names
            })
        else:
            return _databoxes.Databox(**{
                n: [ v.parameters[n] for v in self._variants ]
                for n in parameter_names
            })

    #]

