"""
Implement SimulationPlannableProtocol and SteadyPlannableProtocol
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )

from .. import quantities as _quantities

if TYPE_CHECKING:
    from .main import (Simultaneous, )
#]


class _SimulationPlannable:
    """
    """
    #[

    def __init__(
        self,
        model,
        **kwargs,
    ) -> None:
        """
        """
        curr_xi_qids, *_ = model.solution_vectors.get_curr_transition_indexes()
        qid_to_name = model.create_qid_to_name()
        can_be_exogenized = tuple( qid_to_name[qid] for qid in curr_xi_qids )
        #
        self.can_be_exogenized_anticipated = can_be_exogenized
        self.can_be_exogenized_unanticipated = can_be_exogenized
        #
        def _get_names(kind: _quantities.QuantityKind, ):
            return tuple(_quantities.generate_quantity_names_by_kind(
                model.quantities, kind=kind,
            ))
        #
        self.can_be_endogenized_anticipated \
            = _get_names(_quantities.ANTICIPATED_SHOCK, )
        #
        self.can_be_endogenized_unanticipated \
            = _get_names(_quantities.UNANTICIPATED_SHOCK, )

    #]


class _SteadyPlannable:
    """
    """
    #[

    def __init__(
        self,
        model: Simultaneous,
        is_flat: bool,
        **kwargs,
    ) -> None:
        """
        """
        def _get_names(kind: _quantities.QuantityKind, ):
            return tuple(_quantities.generate_quantity_names_by_kind(
                model.quantities, kind=kind,
            ))
        #
        self.can_be_exogenized = _get_names(_quantities.ENDOGENOUS_VARIABLE, )
        self.can_be_endogenized = _get_names(_quantities.PARAMETER, )
        self.can_be_fixed_level = self.can_be_exogenized
        if is_flat:
            self.can_be_fixed_change = ()
        else:
            self.can_be_fixed_change = self.can_be_exogenized

    #]


class Inlay:
    """
    """
    #[

    def get_simulation_plannable(self, **kwargs, ) -> _SimulationPlannable:
        """
        """
        return _SimulationPlannable(self, **kwargs, )

    def get_steady_plannable(self, **kwargs, ) -> _SteadyPlannable:
        model_flags = self.resolve_flags(**kwargs, )
        return _SteadyPlannable(self, model_flags.is_flat, )

    #]

