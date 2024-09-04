"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

import warnings as _wa
from typing import (TYPE_CHECKING, )

from ..slatables import (Slatable, )
from ..series.main import (Series, )
from .. import quantities as _quantities

if TYPE_CHECKING:
    from typing import (Self, )
    from numbers import (Real, )
#]


class _Slatable(Slatable):
    """
    """
    #[

    @classmethod
    def for_simulate_and_kalman_filter(
        klass,
        model,
        output_kind: _quantities.Quantity,
        **kwargs,
    ) -> Self:
        """
        """
        #
        self = klass(**kwargs, )
        self.max_lag = model.max_lag
        self.max_lead = model.max_lead
        #
        qid_to_name = model.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(qid_to_name)
        )
        #
        name_to_description = model.get_quantity_descriptions()
        self.descriptions = tuple(
            name_to_description.get(name, "", )
            for name in self.databox_names
        )
        #
        # Databox validation - all variables must be time series
        variable_names = model.get_names(kind=_quantities.ANY_VARIABLE, )
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
        #
        parameters = model.get_parameters(unpack_singleton=True, )
        if self.parameters_from_data:
            self.fallbacks.update(parameters, )
        else:
            self.overwrites.update(parameters, )
        #
        shock_names = model.get_names(kind=_quantities.ANY_SHOCK, )
        shock_meds = {
            name: [float(0), ]*model.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_meds, )
        else:
            self.overwrites.update(shock_meds, )
        #
        shock_stds = model.get_stds(unpack_singleton=False, )
        if self.stds_from_data:
            self.fallbacks.update(shock_stds, )
        else:
            self.overwrites.update(shock_stds, )
        #
        self.qid_to_logly = model.create_qid_to_logly()
        self.output_names = model.get_names(kind=output_kind, )
        #
        return self

    @classmethod
    def for_multiply_stds(
        klass,
        model,
        fallbacks: dict[str, list[Real]] | None,
        **kwargs,
    ) -> None:
        """
        """
        #
        self = klass(**kwargs, )
        self.max_lag = 0
        self.max_lead = 0
        #
        # Databox names
        std_qids = _quantities.generate_qids_by_kind(
            model._invariant.quantities,
            kind=_quantities.ANY_STD,
        )
        qid_to_name = model.create_qid_to_name()
        self.databox_names = tuple(
            qid_to_name[qid]
            for qid in sorted(std_qids)
        )
        #
        self.descriptions = None
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
        #
        return self

    #]


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
        output_kind = (
            _quantities.ANY_VARIABLE
            | _quantities.ANY_SHOCK
            | _quantities.UNANTICIPATED_STD
            | _quantities.MEASUREMENT_STD
        )
        slatable = _Slatable.for_simulate_and_kalman_filter(
            self,
            output_kind=output_kind,
            **kwargs,
        )
        #
        return slatable

    def get_slatables_for_multiply_stds(self, **kwargs, ) -> tuple[_Slatable, _Slatable]:
        """
        """
        std_fallbacks = self.get_stds(unpack_singleton=False, )
        std_slatable = _Slatable.for_multiply_stds(self, fallbacks=std_fallbacks, **kwargs, )
        #
        multiplier_fallbacks = { name: 1 for name in std_fallbacks }
        multiplier_slatable = _Slatable.for_multiply_stds(self, fallbacks=multiplier_fallbacks, **kwargs, )
        #
        return std_slatable, multiplier_slatable

    #]

