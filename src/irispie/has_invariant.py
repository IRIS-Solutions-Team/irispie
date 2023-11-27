"""
"""


#[
from __future__ import annotations
#]


class HasInvariantMixin:
    """
    """
    #[

    def get_description(self, /, ) -> str:
        """
        """
        return self._invariant.get_description()

    def set_description(self, *args, **kwargs, ) -> None:
        """
        """
        self._invariant.set_description(*args, **kwargs, )

    #]

