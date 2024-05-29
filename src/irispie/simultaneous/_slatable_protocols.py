"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

from numbers import (Real, )

from .. import quantities as _quantities
from ..series.main import (Series, )
#]


class _Slatable:
    """
    """
    #[

    __slots__ = (
        # Configuration
        "shocks_from_data",
        "stds_from_data",
        # Min and max shifts
        "max_lag",
        "max_lead",
        # Databox names
        "databox_names",
        # Databox validation
        "databox_validators",
        # Fallbacks and overwrites
        "fallbacks",
        "overwrites",
        # QID to logly
        "qid_to_logly",
        # Output names
        "output_names",
    )

    def __init__(
        self,
        shocks_from_data: bool = False,
        stds_from_data: bool = False,
    ) -> None:
        """
        """
        self.shocks_from_data = shocks_from_data
        self.stds_from_data = stds_from_data
        #
        self.max_lag = None
        self.max_lead = None
        self.databox_names = None
        self.databox_validators = None
        self.fallbacks = None
        self.overwrites = None
        self.qid_to_logly = None
        self.output_names = None

    @classmethod
    def for_simulate_and_kalman_filter(
        klass,
        simultaneous,
        output_kind: _quantities.Quantity,
        **kwargs,
    ) -> None:
        """
        """
        #
        self = klass()
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
            "Input data for this variable is not a time series",
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
        shock_names = simultaneous.get_names(kind=_quantities.ANY_SHOCK, )
        shock_meds = {
            name: [float(0), ]*simultaneous.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_meds, )
        else:
            self.overwrites.update(shock_meds, )
        #
        shock_stds = simultaneous.get_stds(unpack_singleton=False, )
        if self.stds_from_data:
            self.fallbacks.update(shock_stds, )
        else:
            self.overwrites.update(shock_stds, )
        #
        self.qid_to_logly = simultaneous.create_qid_to_logly()
        self.output_names = simultaneous.get_names(kind=output_kind, )

    @classmethod
    def for_multiply_stds(
        klass,
        simultaneous,
        fallbacks: dict[str, list[Real]] | None,
        **kwargs,
    ) -> None:
        """
        """
        #
        self = klass()
        self.max_lag = 0
        self.max_lead = 0
        #
        # Databox names
        kind = _quantities.UNANTICIPATED_STD | _quantities.MEASUREMENT_STD
        std_qids = simultaneous.get_qids(kind=kind, )
        qid_to_name = simultaneous.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(std_qids)
        )
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


class Inlay:
    """
    """
    #[

    def get_slatable_for_simulate(self, **kwargs, ) -> _Slatable:
        """
        """
        output_kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
        )
        #
        slatable = _Slatable.for_simulate_and_kalman_filter(
            self,
            output_kind=output_kind,
            **kwargs,
        )
        #
        return slatable

    def get_slatable_for_kalman_filter(self, **kwargs, ) -> _Slatable:
        """
        """
        slatable = _Slatable.for_simulate_and_kalman_filter(self, **kwargs, )
        output_kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
            | _quantities.UNANTICIPATED_STD
            | _quantities.MEASUREMENT_STD
        )
        #
        slatable = _Slatable.for_simulate_and_kalman_filter(
            self,
            output_kind=output_kind,
            **kwargs,
        )
        #
        return slatable

    def get_slatables_for_multiply_shocks(self, **kwargs, ) -> tuple[_Slatable, _Slatable]:
        """
        """
        std_fallbacks = simultaneous.get_stds(unpack_singleton=False, )
        std_slatable = _Slatable.for_multiply_stds(self, fallbacks=std_fallbacks, **kwargs, )
        #
        multiplier_fallbacks = { name: 1 for name in std_fallbacks }
        multiplier_slatable = _Slatable.for_multiply_stds(self, fallbacks=multiplier_fallbacks, **kwargs, )
        #
        return std_slatable, multiplier_slatable

    #]

