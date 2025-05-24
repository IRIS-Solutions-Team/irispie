r"""
Implement SlatableProtocol for Sequential models
"""


#[

from __future__ import annotations

from ..dataslates import Slatable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from .main import Sequential

#]


_DEFAULT_RESIDUAL_VALUE = 0.0


class Inlay:
    r"""
    """
    #[

    def slatable_for_simulate(
        self,
        parameters_from_data: bool,
        shocks_from_data: bool,
    ) -> Slatable:
        """
        """
        slatable = Slatable()
        #
        slatable.max_lag = self.max_lag
        slatable.max_lead = self.max_lead
        #
        slatable.databox_names = self.all_names
        slatable.descriptions = None
        slatable.databox_validators = None
        #
        slatable.fallbacks = {}
        slatable.overwrites = {}
        #
        parameter_name_to_value = self.get_parameters(unpack_singleton=True, )
        if parameters_from_data:
            slatable.fallbacks.update(parameter_name_to_value, )
        else:
            slatable.overwrites.update(parameter_name_to_value, )
        #
        residual_name_to_value = {
            name: _DEFAULT_RESIDUAL_VALUE
            for name in self.residual_names
        }
        if shocks_from_data:
            slatable.fallbacks.update(residual_name_to_value, )
        else:
            slatable.overwrites.update(residual_name_to_value, )
        #
        slatable.qid_to_logly = {}
        #
        slatable.output_names = (
            self.lhs_names
            + self.rhs_only_names
            + self.residual_names
        )
        if parameters_from_data:
            slatable.output_names += self.parameter_names
        #
        return slatable

    #]

