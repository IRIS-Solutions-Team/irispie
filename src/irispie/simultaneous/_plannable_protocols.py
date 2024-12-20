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
    r"""
    ................................................................................
    ==Class: _SimulationPlannable==

    Provides functionality to determine quantities that can be exogenized or 
    endogenized in simulation scenarios. This class handles anticipated and 
    unanticipated shocks based on model configurations.

    Attributes:
        - `can_be_exogenized_anticipated`: Names of quantities that can be exogenized 
          for anticipated shocks.
        - `can_be_exogenized_unanticipated`: Names of quantities that can be exogenized 
          for unanticipated shocks.
        - `can_be_endogenized_anticipated`: Names of quantities that can be endogenized 
          for anticipated shocks.
        - `can_be_endogenized_unanticipated`: Names of quantities that can be endogenized 
          for unanticipated shocks.
    ................................................................................
    """
    #[

    def __init__(
        self,
        model,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Method: __init__==

        Initializes the `_SimulationPlannable` instance with simulation-related 
        quantities. Determines which quantities can be exogenized or endogenized 
        based on the provided model.

        ### Input arguments ###
        ???+ input "model"
            The model object containing quantities and solution vectors.
        ???+ input "**kwargs"
            Additional keyword arguments for future extensibility.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            sim_plannable = _SimulationPlannable(model)
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Class: _SteadyPlannable==

    Provides functionality to determine quantities that can be exogenized, 
    endogenized, or fixed in steady-state scenarios. The configurations depend on 
    whether the model operates in flat mode.

    Attributes:
        - `can_be_exogenized`: Names of quantities that can be exogenized.
        - `can_be_endogenized`: Names of quantities that can be endogenized.
        - `can_be_fixed_level`: Names of quantities that can be fixed at a specific level.
        - `can_be_fixed_change`: Names of quantities that can be fixed at a specific 
          change (dependent on flat mode).
    ................................................................................
    """
    #[

    def __init__(
        self,
        model: Simultaneous,
        is_flat: bool,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Method: __init__==

        Initializes the `_SteadyPlannable` instance with steady-state-related quantities. 
        Configures the fixed-level and fixed-change settings based on the model's flat mode.

        ### Input arguments ###
        ???+ input "model: Simultaneous"
            The model object containing quantities.
        ???+ input "is_flat: bool"
            A flag indicating whether the model operates in flat mode.
        ???+ input "**kwargs"
            Additional keyword arguments for future extensibility.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            steady_plannable = _SteadyPlannable(model, is_flat=True)
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Class: Inlay==

    Acts as a factory for creating instances of `_SimulationPlannable` and 
    `_SteadyPlannable`. Provides access to plannable configurations for simulation 
    and steady-state scenarios.
    ................................................................................
    """
    #[

    def get_simulation_plannable(self, **kwargs, ) -> _SimulationPlannable:
        r"""
        ................................................................................
        ==Method: get_simulation_plannable==

        Creates and returns an instance of `_SimulationPlannable` with configurations 
        for simulation scenarios.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional keyword arguments to pass to `_SimulationPlannable`.

        ### Returns ###
        ???+ returns "_SimulationPlannable"
            An instance of `_SimulationPlannable` configured with simulation-related data.

        ### Example ###
        ```python
            sim_plannable = inlay.get_simulation_plannable()
        ```
        ................................................................................
        """
        return _SimulationPlannable(self, **kwargs, )

    def get_steady_plannable(self, **kwargs, ) -> _SteadyPlannable:
        r"""
        ................................................................................
        ==Method: get_steady_plannable==

        Creates and returns an instance of `_SteadyPlannable` with configurations 
        for steady-state scenarios.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional keyword arguments to resolve model flags.

        ### Returns ###
        ???+ returns "_SteadyPlannable"
            An instance of `_SteadyPlannable` configured with steady-state-related data.

        ### Example ###
        ```python
            steady_plannable = inlay.get_steady_plannable()
        ```
        ................................................................................
        """
        model_flags = self.resolve_flags(**kwargs, )
        return _SteadyPlannable(self, model_flags.is_flat, )

    #]

