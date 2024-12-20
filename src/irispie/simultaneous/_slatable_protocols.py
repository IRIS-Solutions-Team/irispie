"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

import warnings as _wa
from typing import (TYPE_CHECKING, )

from ..slatables import (Slatable, )
from ..series.main import (Series, )
from .. import quantities as _quantities

if TYPE_CHECKING:
    from typing import (Self, )
    from numbers import (Real, )
#]


class _Slatable(Slatable):
    r"""
    ................................................................................
    ==Class: _Slatable==

    Represents a slatable object used in simulations and Kalman filtering. It manages 
    data validation, fallback, and overwrite logic for model-related quantities such 
    as variables, shocks, and parameters.

    Inherits:
        - `Slatable`: Base class for slatable-related operations.

    Attributes:
        - `max_lag`: Maximum lag in the model.
        - `max_lead`: Maximum lead in the model.
        - `databox_names`: Names of the databox variables.
        - `descriptions`: Descriptions of the databox variables.
        - `databox_validators`: Validation rules for databox variables.
        - `fallbacks`: Fallback values for missing data.
        - `overwrites`: Overwrite values for specific data.
        - `qid_to_logly`: Mapping of quantity IDs to log status.
        - `output_names`: Names of the output variables.
    ................................................................................
    """
    #[

    @classmethod
    def for_simulate_and_kalman_filter(
        klass,
        model,
        output_kind: _quantities.Quantity,
        **kwargs,
    ) -> Self:
        r"""
        ................................................................................
        ==Class Method: for_simulate_and_kalman_filter==

        Creates a `_Slatable` instance configured for simulations and Kalman filtering. 
        This method initializes attributes related to data validation, fallback logic, 
        and overwrite logic.

        ### Input arguments ###
        ???+ input "model"
            The model object containing quantities, parameters, and shocks.
        ???+ input "output_kind: _quantities.Quantity"
            The kind of output quantities to include.
        ???+ input "**kwargs"
            Additional parameters for initialization.

        ### Returns ###
        ???+ returns "Self"
            A configured `_Slatable` instance.

        ### Example ###
        ```python
            slatable = _Slatable.for_simulate_and_kalman_filter(model, output_kind=output_quantity)
        ```
        ................................................................................
        """
        #
        self = klass(**kwargs, )
        self.max_lag = model.max_lag
        self.max_lead = model.max_lead
        #
        qid_to_name = model.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(qid_to_name)
        )
        #
        name_to_description = model.get_quantity_descriptions()
        self.descriptions = tuple(
            name_to_description.get(name, "", )
            for name in self.databox_names
        )
        #
        # Databox validation - all variables must be time series
        variable_names = model.get_names(kind=_quantities.ANY_VARIABLE, )
        validator = (
            lambda x: isinstance(x, Series),
            "Input data for this variable is not a time series",
        )
        self.databox_validators = {
            name: validator
            for name in variable_names
        }
        #
        # Fallbacks and overwrites
        self.fallbacks = {}
        self.overwrites = {}
        #
        #
        parameters = model.get_parameters(unpack_singleton=True, )
        if self.parameters_from_data:
            self.fallbacks.update(parameters, )
        else:
            self.overwrites.update(parameters, )
        #
        shock_names = model.get_names(kind=_quantities.ANY_SHOCK, )
        shock_meds = {
            name: [float(0), ]*model.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_meds, )
        else:
            self.overwrites.update(shock_meds, )
        #
        shock_stds = model.get_stds(unpack_singleton=False, )
        if self.stds_from_data:
            self.fallbacks.update(shock_stds, )
        else:
            self.overwrites.update(shock_stds, )
        #
        self.qid_to_logly = model.create_qid_to_logly()
        self.output_names = model.get_names(kind=output_kind, )
        #
        return self

    @classmethod
    def for_multiply_stds(
        klass,
        model,
        fallbacks: dict[str, list[Real]] | None,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Class Method: for_multiply_stds==

        Creates a `_Slatable` instance configured for multiplying standard deviations. 
        This method focuses on managing fallback logic for standard deviation data.

        ### Input arguments ###
        ???+ input "model"
            The model object containing quantities and standard deviations.
        ???+ input "fallbacks: dict[str, list[Real]] | None"
            Fallback values for standard deviations.
        ???+ input "**kwargs"
            Additional parameters for initialization.

        ### Returns ###
        ???+ returns "Self"
            A configured `_Slatable` instance.

        ### Example ###
        ```python
            slatable = _Slatable.for_multiply_stds(model, fallbacks=fallback_data)
        ```
        ................................................................................
        """
        #
        self = klass(**kwargs, )
        self.max_lag = 0
        self.max_lead = 0
        #
        # Databox names
        std_qids = _quantities.generate_qids_by_kind(
            model._invariant.quantities,
            kind=_quantities.ANY_STD,
        )
        qid_to_name = model.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(std_qids)
        )
        #
        self.descriptions = None
        #
        # Databox validation - all variables must be time series
        self.databox_validators = None
        #
        # Fallbacks and overwrites
        self.fallbacks = fallbacks
        self.overwrites = None
        #
        self.qid_to_logly = None
        self.output_names = self.databox_names
        #
        return self

    #]


class Inlay:
    r"""
    ................................................................................
    ==Class: Inlay==

    Provides methods to create and manage `_Slatable` objects for simulations, 
    Kalman filtering, and operations involving standard deviations. This class 
    encapsulates the logic for setting up and managing slatable configurations.

    Attributes:
        - None (Attributes are dynamically managed during method execution).
    ................................................................................
    """
    #[

    def get_slatable_for_simulate(self, **kwargs, ) -> _Slatable:
        r"""
        ................................................................................
        ==Method: get_slatable_for_simulate==

        Creates a `_Slatable` object configured for simulations. The slatable is set up 
        to manage variables and shocks for the simulation process.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional parameters passed to `_Slatable` initialization.

        ### Returns ###
        ???+ returns "_Slatable"
            A configured `_Slatable` instance for simulations.

        ### Example ###
        ```python
            slatable = obj.get_slatable_for_simulate()
        ```
        ................................................................................
        """
        output_kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
        )
        slatable = _Slatable.for_simulate_and_kalman_filter(
            self,
            output_kind=output_kind,
            **kwargs,
        )
        #
        return slatable

    def get_slatable_for_kalman_filter(self, **kwargs, ) -> _Slatable:
        r"""
        ................................................................................
        ==Method: get_slatable_for_kalman_filter==

        Creates a `_Slatable` object configured for Kalman filtering. The slatable 
        includes variables, shocks, and additional statistical quantities like 
        measurement standard deviations.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional parameters passed to `_Slatable` initialization.

        ### Returns ###
        ???+ returns "_Slatable"
            A configured `_Slatable` instance for Kalman filtering.

        ### Example ###
        ```python
            slatable = obj.get_slatable_for_kalman_filter()
        ```
        ................................................................................
        """
        output_kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
            | _quantities.UNANTICIPATED_STD
            | _quantities.MEASUREMENT_STD
        )
        slatable = _Slatable.for_simulate_and_kalman_filter(
            self,
            output_kind=output_kind,
            **kwargs,
        )
        #
        return slatable

    def get_slatables_for_multiply_stds(self, **kwargs, ) -> tuple[_Slatable, _Slatable]:
        r"""
        ................................................................................
        ==Method: get_slatables_for_multiply_stds==

        Creates a pair of `_Slatable` objects for operations involving standard 
        deviations. One slatable handles standard deviation data, while the other 
        handles multiplier data for scaling.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional parameters passed to `_Slatable` initialization.

        ### Returns ###
        ???+ returns "tuple[_Slatable, _Slatable]"
            A tuple containing two `_Slatable` instances:
            - The first for standard deviation data.
            - The second for multiplier data.

        ### Example ###
        ```python
            std_slatable, multiplier_slatable = obj.get_slatables_for_multiply_stds()
        ```
        ................................................................................
        """
        std_fallbacks = self.get_stds(unpack_singleton=False, )
        std_slatable = _Slatable.for_multiply_stds(self, fallbacks=std_fallbacks, **kwargs, )
        #
        multiplier_fallbacks = { name: 1 for name in std_fallbacks }
        multiplier_slatable = _Slatable.for_multiply_stds(self, fallbacks=multiplier_fallbacks, **kwargs, )
        #
        return std_slatable, multiplier_slatable

    #]

