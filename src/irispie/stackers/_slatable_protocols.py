r"""
Implement SlatableProtocol for Stacker objects
"""


#[

from __future__ import annotations

from ..dataslates import Slatable
from ..series import Series
from .. import quantities as _quantities

#]


_DEFAULT_SHOCK_VALUE = 0.0


class Inlay:
    r"""
    """
    #[

    def slatable_for_marginal(
        self,
        shocks_from_data: bool,
        stds_from_data: bool,
    ) -> Slatable:
        r"""
        """
        #
        if self.num_variants != 1:
            raise ValueError("Not implemented for multiple variants")
        #
        slatable = Slatable()
        #
        slatable.max_lag = 0 # self.max_lag
        slatable.max_lead = 0 # self.max_lead
        #
        qid_to_name = self.create_qid_to_name()
        slatable.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(qid_to_name)
        )
        #
        name_to_description = self.create_name_to_description()
        slatable.descriptions = tuple(
            name_to_description.get(name, "", )
            for name in slatable.databox_names
        )
        #
        # Databox validation - all variables must be time series
        variable_names = self.get_names(kind=_quantities.ANY_VARIABLE, )
        validator = (
            lambda x: isinstance(x, Series),
            "Input data for this variable is not a time series",
        )
        slatable.databox_validators = {
            name: validator
            for name in variable_names
        }
        #
        # Fallbacks and overwrites
        slatable.fallbacks = {}
        slatable.overwrites = {}
        #
        shock_names = self.get_names(kind=_quantities.ANY_SHOCK, )
        shock_name_to_value = {
            name: [_DEFAULT_SHOCK_VALUE, ]*self.num_variants
            for name in shock_names
        }
        if shocks_from_data:
            slatable.fallbacks.update(shock_name_to_value, )
        else:
            slatable.overwrites.update(shock_name_to_value, )
        #
        stds = self._variants[0].std_name_to_value
        if stds_from_data:
            slatable.fallbacks.update(stds, )
        else:
            slatable.overwrites.update(stds, )
        #
        slatable.qid_to_logly = self.create_qid_to_logly()
        #
        return slatable

    #]

