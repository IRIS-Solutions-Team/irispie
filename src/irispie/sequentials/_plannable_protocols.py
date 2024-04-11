"""
Implement SimulationPlannableProtocol
"""


#[
from __future__ import annotations
#]


class _SimulationPlannable:
    """
    """
    #[

    def __init__(
        self,
        sequential,
        **kwargs,
    ) -> None:
        """
        """
        self.can_be_exogenized \
            = tuple(set(
                i.lhs_name for i in sequential._invariant.explanatories
                if not i.is_identity
            ))
        self.can_be_endogenized \
            = tuple(set(
                i.residual_name for i in sequential._invariant.explanatories
                if not i.is_identity
            ))


class Inlay:
    """
    """
    #[

    def get_simulation_plannable(self, **kwargs, ) -> _SimulationPlannable:
        """
        """
        return _SimulationPlannable(self, **kwargs, )

    #]

