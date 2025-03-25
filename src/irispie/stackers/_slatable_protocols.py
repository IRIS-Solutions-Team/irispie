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


class Slatable(_slatables.Slatable):
    """
    """
    #[

    def __init__(
        self,
        stacker: Stacker,
        **kwargs,
    ) -> Self:
        """
        """
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
        parameters = stacker.get_parameters(unpack_singleton=True, )
        self.overwrites.update(parameters, )
        #
        shock_names = stacker.get_names(kind=_quantities.ANY_SHOCK, )
        shock_meds = {
            name: [float(0), ]*stacker.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_meds, )
        else:
            self.overwrites.update(shock_meds, )
        #
        # shock_stds = stacker.get_stds(unpack_singleton=False, )
        # if self.stds_from_data:
        #     self.fallbacks.update(shock_stds, )
        # else:
        #     self.overwrites.update(shock_stds, )
        #
        self.qid_to_logly = stacker.create_qid_to_logly()

    #]

