"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
#]


class Slatable:
    """
    """
    #[

    def __init__(
        self,
        simultaneous,
        *,
        shocks_from_data: bool = False,
        stds_from_data: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        #
        # Min and max shifts
        self.max_lag = simultaneous.max_lag
        self.max_lead = simultaneous.max_lead
        #
        # Databox names
        qid_to_name = simultaneous.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(qid_to_name)
        )
        #
        # Fallbacks and overwrites
        self.fallbacks = {}
        self.overwrites = simultaneous.get_parameters(unpack_singleton=False, )
        #
        num_variants = simultaneous.num_variants
        shock_names = simultaneous.get_names(kind=_quantities.ANY_SHOCK, )
        shock_meds = { name: [float(0), ]*num_variants for name in shock_names }
        shock_stds = simultaneous.get_stds(unpack_singleton=False, )
        #
        if shocks_from_data:
            self.fallbacks.update(shock_meds, )
        else:
            self.overwrites.update(shock_meds, )
        #
        if stds_from_data:
            self.fallbacks.update(shock_stds, )
        else:
            self.overwrites.update(shock_stds, )
        #
        # Qid to logly
        self.qid_to_logly = simultaneous.create_qid_to_logly()
        #
        # Output names
        kind = _quantities.ANY_VARIABLE | _quantities.ANY_SHOCK
        self.output_names = simultaneous.get_names(kind=kind, )


class Inlay:
    """
    """
    #[

    def get_slatable(self, **kwargs, ) -> Slatable:
        """
        """
        return Slatable(self, **kwargs, )


