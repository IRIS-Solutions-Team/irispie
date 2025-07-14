r"""
Implement SlatableProtocol for RedVAR models
"""


#[

from __future__ import annotations

import warnings as _wa

from ..series import Series
from ..dataslates import Slatable
from .. import quantities as _quantities

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from numbers import Real
    from .red_vars import RedVAR

#]


_DEFAULT_RESIDUAL_VALUE = 0.0


class Inlay:
    r"""
    """
    #[

    def slatable_for_estimate(self, **kwargs, ) -> Slatable:
        r"""
        """
        slatable = _slatable_for_anything(self, )
        residual_name_to_value = _create_residual_name_to_value(self, )
        slatable.overwrites.update(residual_name_to_value, )
        return slatable

    def slatable_for_simulate(
        self,
        residuals_from_data: bool,
        **kwargs,
    ) -> Slatable:
        r"""
        """
        slatable = _slatable_for_anything(self, )
        residual_name_to_value = _create_residual_name_to_value(self, )
        if residuals_from_data:
            slatable.fallbacks.update(residual_name_to_value, )
        else:
            slatable.overwrites.update(residual_name_to_value, )
        return slatable

    slatable_for_kalman_filter = slatable_for_simulate

    #]


def _slatable_for_anything(model: RedVAR, ) -> Slatable:
    r"""
    Create slatable base for estimate or simulate contexts
    """
    #{
    slatable = Slatable()
    #
    slatable.max_lag = -model.order
    #
    slatable.databox_names = model.get_names()
    slatable.output_names = slatable.databox_names
    #
    name_to_description = model.create_name_to_description()
    slatable.descriptions = tuple(
        name_to_description.get(name, "", )
        for name in slatable.databox_names
    )
    #
    # Fallbacks and overwrites
    slatable.fallbacks = {}
    slatable.overwrites = {}
    #
    # Databox validation - all input data must be time series
    validator = (
        lambda x: isinstance(x, Series),
        "Input data for this variable is not a time series",
    )
    slatable.databox_validators = {
        name: validator
        for name in slatable.databox_names
    }
    #
    return slatable
    #]


def _create_residual_name_to_value(model, ) -> dict[str, Real]:
    r"""
    Create dict with default residual values for all residuals in the model
    """
    #[
    residual_names = model.get_names(kind=_quantities.UNANTICIPATED_SHOCK, )
    return {
        name: _DEFAULT_RESIDUAL_VALUE
        for name in residual_names
    }
    #]

