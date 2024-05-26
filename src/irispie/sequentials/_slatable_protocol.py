"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations
#]


class _Slatable:
    """
    """
    #[

    def __init__(
        self,
        sequential,
        *,
        shocks_from_data: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        #
        self.max_lag = sequential.max_lag
        self.max_lead = sequential.max_lead
        #
        self.databox_names = sequential.all_names
        self.databox_validators = None
        #
        residual_names = tuple(sequential._invariant.residual_names)
        residuals = { name: 0 for name in residual_names }
        #
        self.fallbacks = {}
        self.overwrites = sequential.get_parameters(unpack_singleton=False, )
        #
        if shocks_from_data:
            self.fallbacks.update(residuals, )
        else:
            self.overwrites.update(residuals, )
        #
        self.qid_to_logly = {}
        #
        self.output_names = (
            sequential.lhs_names
            + sequential.rhs_only_names
            + sequential.residual_names
        )


class Inlay:
    """
    """
    #[

    def get_slatable_for_simulation(self, **kwargs, ) -> _Slatable:
        """
        """
        return _Slatable(self, **kwargs, )


