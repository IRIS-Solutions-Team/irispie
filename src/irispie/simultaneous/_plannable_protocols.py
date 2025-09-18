"""
Implement SimulationPlannableProtocol and SteadyPlannableProtocol
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )

from .. import quantities as _quantities

from ..quantities import (
    TRANSITION_VARIABLE,
    TRANSITION_SHOCK,
    ANTICIPATED_SHOCK_VALUE,
    ENDOGENOUS_VARIABLE,
    PARAMETER,
)

if TYPE_CHECKING:
    from .main import Simultaneous
    from ..quantities import QuantityKind
#]


#-------------------------------------------------------------------------------
# Functions to be used as methods in Simultaneous class
#-------------------------------------------------------------------------------


def inlay(klass: type, ) -> type:
    r"""
    Inlay plannable protocol methods in the class
    """
    #[
    klass.get_simulation_plannable = get_simulation_plannable
    klass.get_steady_plannable = get_steady_plannable
    return klass
    #]


def get_simulation_plannable(self, **kwargs, ) -> _SimulationPlannable:
    """
    """
    return _SimulationPlannable(model=self, **kwargs, )


def get_steady_plannable(self, **kwargs, ) -> _SteadyPlannable:
    model_flags = self.resolve_flags(**kwargs, )
    return _SteadyPlannable(model=self, is_flat=model_flags.is_flat, )


#-------------------------------------------------------------------------------


class _SimulationPlannable:
    r"""
    """
    #[

    def __init__(
        self,
        model,
        **kwargs,
    ) -> None:
        r"""
        """
        #
        def get_names(kind: QuantityKind, ):
            return tuple(_quantities.generate_quantity_names_by_kind(
                model.quantities, kind=kind,
            ))
        #
        # qid_to_name = model.create_qid_to_name()
        # curr_xi_qids, *_ = model.solution_vectors.get_curr_transition_indexes()
        # can_be_exogenized = tuple( qid_to_name[qid] for qid in curr_xi_qids )
        #
        can_be_exogenized = get_names(TRANSITION_VARIABLE, )
        self.can_be_exogenized_anticipated = can_be_exogenized
        self.can_be_exogenized_unanticipated = can_be_exogenized
        self.can_be_endogenized_unanticipated = get_names(TRANSITION_SHOCK, )
        self.can_be_endogenized_anticipated = get_names(ANTICIPATED_SHOCK_VALUE, )

    #]


class _SteadyPlannable:
    r"""
    """
    #[

    def __init__(
        self,
        model: Simultaneous,
        is_flat: bool,
        **kwargs,
    ) -> None:
        r"""
        """
        def get_names(kind: QuantityKind, ):
            return tuple(_quantities.generate_quantity_names_by_kind(
                model.quantities, kind=kind,
            ))
        #
        self.can_be_exogenized = get_names(ENDOGENOUS_VARIABLE, )
        # self.can_be_exogenized = get_names(TRANSITION_VARIABLE, )
        self.can_be_endogenized = get_names(PARAMETER, )
        self.can_be_fixed_level = self.can_be_exogenized
        self.can_be_fixed_change = ()
        if not is_flat:
            self.can_be_fixed_change = self.can_be_exogenized

    #]

