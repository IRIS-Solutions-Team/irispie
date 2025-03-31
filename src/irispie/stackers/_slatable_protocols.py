"""
Implement SlatableProtocol
"""


#[

from __future__ import annotations

import warnings as _wa

from .. import slatables as _slatables
from ..series.main import Series
from .. import quantities as _quantities

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from .main import Stacker

#]


_DEFAULT_SHOCK_VALUE = 0.0


class Slatable(_slatables.Slatable):
    r"""
    """
    #[

    def __init__(
        self,
        stacker: Stacker,
        **kwargs,
    ) -> Self:
        r"""
        """
        #
        if stacker.num_variants != 1:
            raise ValueError("Not implemented for multiple variants")
        #
        super().__init__(**kwargs, )
        self.max_lag = stacker.max_lag
        self.max_lead = stacker.max_lead
        #
        qid_to_name = stacker.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(qid_to_name)
        )
        #
        name_to_description = stacker.create_name_to_description()
        self.descriptions = tuple(
            name_to_description.get(name, "", )
            for name in self.databox_names
        )
        #
        # Databox validation - all variables must be time series
        variable_names = stacker.get_names(kind=_quantities.ANY_VARIABLE, )
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
        shock_names = stacker.get_names(kind=_quantities.ANY_SHOCK, )
        shock_name_to_value = {
            name: [_DEFAULT_SHOCK_VALUE, ]*stacker.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_name_to_value, )
        else:
            self.overwrites.update(shock_name_to_value, )
        #
        stds = stacker._variants[0].std_name_to_value
        if self.stds_from_data:
            self.fallbacks.update(stds, )
        else:
            self.overwrites.update(stds, )
        #
        self.qid_to_logly = stacker.create_qid_to_logly()

    #]

