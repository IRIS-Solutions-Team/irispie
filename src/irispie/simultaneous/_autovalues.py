"""
"""

#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )
import numpy as _np

from .. import equations as _equations

if TYPE_CHECKING:
    from typing import (Iterable, )
#]

class Inlay:
    """
    """
    #[

    def update_autovalues(self, ) -> None:
        """
        Verify currently assigned steady state in dynamic or steady equations for each variant within this model
        """
        qid_to_logly = self.create_qid_to_logly()
        num_columns = self.max_lead - self.min_lag + 1
        shift_in_first_column = -self.min_lag
        autovalue_definitions = _equations.generate_equations_of_kind(
            self._invariant.dynamic_equations,
            kind=_equations.EquationKind.AUTOVALUE_DEFINITION,
        )
        #
        for variant in self._variants:
            steady_array = variant.create_steady_array(
                qid_to_logly,
                num_columns=num_columns,
                shift_in_first_column=shift_in_first_column,
            )
            import ipdb; ipdb.set_trace()
            _calculate_autovalues(autovalue_definitions, steady_array, )

    #]


def _calculate_autovalues(
    autovalue_definitions: Iterable[_equations.Equation],
    steady_array: _np.ndarray,
    /,
) -> None:
    """
    """
    pass

