"""
Implement SimulationPlannableProtocol and SteadyPlannableProtocol
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
#]


_SIMULATE_CAN_BE_EXOGENIZED = _quantities.QuantityKind.ENDOGENOUS_VARIABLE
_SIMULATE_CAN_BE_ENDOGENIZED = _quantities.QuantityKind.EXOGENOUS_VARIABLE | _quantities.QuantityKind.ANY_SHOCK
_SIMULATE_CAN_BE_EXOGENIZED_ANTICIPATED = _quantities.QuantityKind.ENDOGENOUS_VARIABLE
_SIMULATE_CAN_BE_ENDOGENIZED_ANTICIPATED = _quantities.QuantityKind.ANTICIPATED_SHOCK
_SIMULATE_CAN_BE_EXOGENIZED_UNANTICIPATED = _quantities.QuantityKind.ENDOGENOUS_VARIABLE
_SIMULATE_CAN_BE_ENDOGENIZED_UNANTICIPATED = _quantities.QuantityKind.UNANTICIPATED_SHOCK


class _SimulationPlannable:
    """
    """
    #[

    def __init__(
        self,
        simultaneous,
        **kwargs,
    ) -> None:
        """
        """
        self.can_be_exogenized_anticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                simultaneous._invariant.quantities,
                _SIMULATE_CAN_BE_EXOGENIZED_ANTICIPATED,
            ))
        self.can_be_exogenized_unanticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                simultaneous._invariant.quantities,
                _SIMULATE_CAN_BE_EXOGENIZED_UNANTICIPATED,
            ))
        self.can_be_endogenized_anticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                simultaneous._invariant.quantities,
                _SIMULATE_CAN_BE_ENDOGENIZED_ANTICIPATED,
            ))
        self.can_be_endogenized_unanticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                simultaneous._invariant.quantities,
                _SIMULATE_CAN_BE_ENDOGENIZED_UNANTICIPATED,
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

