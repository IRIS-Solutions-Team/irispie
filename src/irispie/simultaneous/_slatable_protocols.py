r"""
Implement SlatableProtocol for Simultaneous models
"""


#[

from __future__ import annotations

from ..series.main import Series
from .. import quantities as _quantities
from ..dataslates import Slatable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Real

#]


_DEFAULT_SHOCK_VALUE = 0.0


_OUTPUT_KIND_FOR_SIMULATE = (
    _quantities.ANY_VARIABLE
    | _quantities.ANY_SHOCK
)


_OUTPUT_KIND_FOR_KALMAN_FILTER = (
    _quantities.ANY_VARIABLE
    | _quantities.ANY_SHOCK
    | _quantities.UNANTICIPATED_STD
    | _quantities.MEASUREMENT_STD
)


class Inlay:
    r"""
    """
    #[

    def slatable_for_simulate(self, **kwargs, ) -> Slatable:
        r"""
        """
        slatable = _slatable_for_simulate_or_kalman_filter(self, **kwargs, )
        output_kind = _OUTPUT_KIND_FOR_SIMULATE
        if kwargs.get("output_parameters", False):
            output_kind |= _quantities.PARAMETER
        slatable.output_names = self.get_names(kind=output_kind, )
        return slatable

    def slatable_for_kalman_filter(self, **kwargs, ) -> Slatable:
        r"""
        """
        slatable = _slatable_for_simulate_or_kalman_filter(self, parameters_from_data=False, **kwargs, )
        output_kind = _OUTPUT_KIND_FOR_KALMAN_FILTER
        if kwargs.get("output_parameters", False):
            output_kind |= _quantities.PARAMETER
        slatable.output_names = self.get_names(kind=output_kind, )
        return slatable

    def slatables_for_vary_stds(self, ) -> tuple[Slatable, Slatable]:
        r"""
        """
        std_slatable = _slatable_for_vary_stds(self, )
        std_name_to_value = self.get_stds(unpack_singleton=False, )
        std_slatable.fallbacks = std_name_to_value
        #
        multiplier_slatable = _slatable_for_vary_stds(self, )
        multiplier_name_to_value = { name: 1 for name in std_name_to_value.keys() }
        multiplier_slatable.fallbacks = multiplier_name_to_value
        #
        return std_slatable, multiplier_slatable,

    #]


def _slatable_for_simulate_or_kalman_filter(
    model,
    parameters_from_data: bool,
    shocks_from_data: bool,
    stds_from_data: bool,
    **kwargs,
) -> Slatable:
    r"""
    """
    #[
    slatable = Slatable()
    #
    slatable.max_lag = model.max_lag
    slatable.max_lead = model.max_lead
    #
    qid_to_name = model.create_qid_to_name()
    slatable.databox_names = tuple(
        qid_to_name[qid]
        for qid in sorted(qid_to_name)
    )
    #
    name_to_description = model.create_name_to_description()
    slatable.descriptions = tuple(
        name_to_description.get(name, "", )
        for name in slatable.databox_names
    )
    #
    # Databox validation - all variables must be time series
    variable_names = model.get_names(kind=_quantities.ANY_VARIABLE, )
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
    parameter_name_to_value = model.get_parameters(unpack_singleton=True, )
    if parameters_from_data:
        slatable.fallbacks.update(parameter_name_to_value, )
    else:
        slatable.overwrites.update(parameter_name_to_value, )
    #
    shock_names = model.get_names(kind=_quantities.ANY_SHOCK, )
    shock_name_to_value = {
        name: [_DEFAULT_SHOCK_VALUE, ]*model.num_variants
        for name in shock_names
    }
    if shocks_from_data:
        slatable.fallbacks.update(shock_name_to_value, )
    else:
        slatable.overwrites.update(shock_name_to_value, )
    #
    std_name_to_value = model.get_stds(unpack_singleton=False, )
    if stds_from_data:
        slatable.fallbacks.update(std_name_to_value, )
    else:
        slatable.overwrites.update(std_name_to_value, )
    #
    slatable.qid_to_logly = model.create_qid_to_logly()
    #
    return slatable
    #]


def _slatable_for_vary_stds(model, ) -> Slatable:
    r"""
    """
    #[
    slatable = Slatable()
    std_qids = _quantities.generate_qids_by_kind(
        model._invariant.quantities,
        kind=_quantities.ANY_STD,
    )
    qid_to_name = model.create_qid_to_name()
    slatable.databox_names = tuple(
        qid_to_name[qid]
        for qid in sorted(std_qids)
    )
    slatable.output_names = slatable.databox_names
    return slatable
    #]

