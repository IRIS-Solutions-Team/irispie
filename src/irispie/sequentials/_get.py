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
        r"""
        """
        return tuple(self.parameter_names)

    def get_residual_names(self, /, ) -> tuple[str, ...]:
        r"""
        """
        return tuple(self.residual_names)

    def get_equations(self, /, ) -> tuple[str, ...]:
        r"""
        """
        return tuple(e.human for e in self._invariant.equations)

    #]

