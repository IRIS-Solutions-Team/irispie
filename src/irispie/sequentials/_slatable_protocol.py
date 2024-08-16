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
        residual_names = tuple(sequential._invariant.residual_names)
        residuals = { name: 0 for name in residual_names }
        #
        #
        #
        self.fallbacks = {}
        self.overwrites = {}
        #
        #
        parameters = sequential.get_parameters(unpack_singleton=True, )
        if self.parameters_from_data:
            self.fallbacks.update(parameters, )
        else:
            self.overwrites.update(parameters, )
        #
        #
        if self.shocks_from_data:
            self.fallbacks.update(residuals, )
        else:
            self.overwrites.update(residuals, )
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


