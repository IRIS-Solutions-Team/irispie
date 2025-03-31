"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )

from ..slatables import (Slatable, )

if TYPE_CHECKING:
    from typing import (Self, )
#]


_DEFAULT_RESIDUAL_VALUE = 0.0


class _Slatable(Slatable):
    """
    """
    #[

    @classmethod
    def for_simulate(
        klass,
        sequential,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass(**kwargs, )
        #
        self.max_lag = sequential.max_lag
        self.max_lead = sequential.max_lead
        #
        self.databox_names = sequential.all_names
        self.descriptions = None
        self.databox_validators = None
        #
        #
        self.fallbacks = {}
        self.overwrites = {}
        #
        #
        parameter_name_to_value = sequential.get_parameters(unpack_singleton=True, )
        if self.parameters_from_data:
            self.fallbacks.update(parameter_name_to_value, )
        else:
            self.overwrites.update(parameter_name_to_value, )
        #
        #
        residual_name_to_value = {
            name: _DEFAULT_RESIDUAL_VALUE
            for name in sequential.residual_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(residual_name_to_value, )
        else:
            self.overwrites.update(residual_name_to_value, )
        #
        #
        self.qid_to_logly = {}
        #
        self.output_names = (
            sequential.lhs_names
            + sequential.rhs_only_names
            + sequential.residual_names
        )
        if self.parameters_from_data:
            self.output_names += sequential.parameter_names
        #
        return self


class Inlay:
    """
    """
    #[

    def get_slatable_for_simulation(self, **kwargs, ) -> _Slatable:
        """
        """
        return _Slatable.for_simulate(self, **kwargs, )


