"""
Implement SimulationPlannableProtocol and SteadyPlannableProtocol
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
#]


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
        quantities = simultaneous.quantities
        solution_vectors = simultaneous.solution_vectors
        #
        curr_xi_qids, *_ = solution_vectors.get_curr_transition_indexes()
        qid_to_name = simultaneous.create_qid_to_name()
        can_be_exogenized = tuple( qid_to_name[qid] for qid in curr_xi_qids )
        #
        self.can_be_exogenized_anticipated = can_be_exogenized
        self.can_be_exogenized_unanticipated = can_be_exogenized
        #
        self.can_be_endogenized_anticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                quantities, _quantities.ANTICIPATED_SHOCK,
            ))
        #
        self.can_be_endogenized_unanticipated \
            = tuple(_quantities.generate_quantity_names_by_kind(
                quantities, _quantities.UNANTICIPATED_SHOCK,
            ))

    #]


class Inlay:
    """
    """
    #[

    def get_simulation_plannable(self, **kwargs, ) -> _SimulationPlannable:
        """
        """
        return _SimulationPlannable(self, **kwargs, )

    #]

