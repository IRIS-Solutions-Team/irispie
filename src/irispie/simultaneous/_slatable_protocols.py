"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
from ..series.main import (Series, )
#]


class _Slatable:
    """
    """
    #[

    def __init__(
        self,
        simultaneous,
        output_names: Iterable[str],
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
        # Databox validation - all variables must be time series
        variable_names = simultaneous.get_names(kind=_quantities.ANY_VARIABLE, )
        validator = (
            lambda x: isinstance(x, Series),
            "Data for this variable is not a time series",
        )
        self.databox_validators = {
            name: validator
            for name in variable_names
        }
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
        self.qid_to_logly = simultaneous.create_qid_to_logly()
        self.output_names = output_names


class Inlay:
    """
    """
    #[

    def get_slatable_for_simulate(self, **kwargs, ) -> _Slatable:
        """
        """
        kind = _quantities.ANY_VARIABLE | _quantities.ANY_SHOCK
        output_names = self.get_names(kind=kind, )
        return _Slatable(self, output_names, **kwargs, )

    def get_slatable_for_kalman_filter(self, **kwargs, ) -> _Slatable:
        """
        """
        kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
            | _quantities.UNANTICIPATED_STD
            | _quantities.MEASUREMENT_STD
        )
        output_names = self.get_names(kind=kind, )
        return _Slatable(self, output_names, **kwargs, )

    #]

