"""
"""


#[
from __future__ import absolute_import
#]


class Inlay:
    """
    """
    #[

    def get_parameter_names(self, /, ) -> tuple[str, ...]:
        """
        """
        return tuple(self._invariant.parameter_names)

    def get_equations(self, /, ) -> tuple[str, ...]:
        """
        """
        return tuple(e.human for e in self._invariant.equations)

    #]

