"""
Implement SlatableProtocol
"""


#[
from __future__ import annotations

import warnings as _wa
from typing import (TYPE_CHECKING, )

from ..slatables import Slatable
from ..series.main import Series
from .. import quantities as _quantities

if TYPE_CHECKING:
    from typing import (Self, )
    from numbers import (Real, )
#]


_DEFAULT_SHOCK_VALUE = 0.0


class _Slatable(Slatable, ):
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
        r"""
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
        name_to_description = model.create_name_to_description()
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
        parameter_name_to_value = model.get_parameters(unpack_singleton=True, )
        if self.parameters_from_data:
            self.fallbacks.update(parameter_name_to_value, )
        else:
            self.overwrites.update(parameter_name_to_value, )
        #
        shock_names = model.get_names(kind=_quantities.ANY_SHOCK, )
        shock_name_to_value = {
            name: [_DEFAULT_SHOCK_VALUE, ]*model.num_variants
            for name in shock_names
        }
        if self.shocks_from_data:
            self.fallbacks.update(shock_name_to_value, )
        else:
            self.overwrites.update(shock_name_to_value, )
        #
        std_name_to_value = model.get_stds(unpack_singleton=False, )
        if self.stds_from_data:
            self.fallbacks.update(std_name_to_value, )
        else:
            self.overwrites.update(std_name_to_value, )
        #
        self.qid_to_logly = model.create_qid_to_logly()
        self.output_names = model.get_names(kind=output_kind, )
        #
        return self

    @classmethod
    def for_multiply_stds(
        klass,
        model,
        fallback_name_to_value: dict[str, list[Real]] | None,
        **kwargs,
    ) -> None:
        r"""
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
        self.fallbacks = fallback_qid_to_value
        self.overwrites = None
        #
        self.qid_to_logly = None
        self.output_names = self.databox_names
        #
        return self

    #]


class Inlay:
    r"""
    """
    #[

    def get_slatable_for_simulate(self, **kwargs, ) -> _Slatable:
        r"""
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
        r"""
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
        r"""
        """
        std_name_to_value = self.get_stds(unpack_singleton=False, )
        std_slatable = _Slatable.for_multiply_stds(
            self,
            fallback_name_to_value=std_name_to_value,
            **kwargs,
        )
        #
        multiplier_name_to_value = { name: 1 for name in std_name_to_value.keys() }
        multiplier_slatable = _Slatable.for_multiply_stds(
            self,
            fallback_name_to_value=multiplier_name_to_value,
            **kwargs,
        )
        #
        return std_slatable, multiplier_slatable

    #]

