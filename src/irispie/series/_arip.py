"""
"""


#[
from __future__ import annotations

from typing import (Self, Iterable, Literal, )
from numbers import (Real, )
import functools as _ft
import numpy as _np

from .. import dates as _dates
from . import _conversions as _conversions
#]

FormType = Literal["rate", "diff", ]
AggregationType = Literal["sum", "mean", "avg", "last", "first", ]

def disaggregate_arip(
    self,
    target_dater_class: _dates.Dater,
    /,
    model: tuple[FormType, AggregationType],
    indicator: _np.ndarray | None = None,
    target_level: Self | None = None,
) -> Self:
    """
    Autoregressive interpolation
    """
    #[
    if not self.start_date:
        return None, None

    low_freq = self.start_date.frequency
    num_low_periods = self.num_periods
    high_freq = target_dater_class.frequency
    num_within = high_freq // low_freq
    num_high_periods = num_low_periods * num_within
    high_start_date = self.start_date.convert(high_freq, position="start", )
    high_end_date = self.end_date.convert(high_freq, position="end", )

    form_string, aggregation_string = model
    form = _CHOOSE_FORM[form_string]

    create_multiplier_column = _ft.partial(
        _create_multiplier_column,
        num_low_periods=num_low_periods,
        num_within=num_within,
    )

    create_target_level_column = _ft.partial(
        _create_target_level_column,
        num_high_periods=num_high_periods,
    )

    create_aggregation_row = _ft.partial(
        _create_aggregation_row,
        num_low_periods=num_low_periods,
        num_within=num_within,
        aggregation_vector=_CHOOSE_AGGREGATION[aggregation_string](num_within, ),
    )

    create_target_level_row = _ft.partial(
        _create_target_level_row,
        num_high_periods=num_high_periods,
    )

    target_level_data, where_finite_target = \
        _get_target_info(target_level, high_start_date, high_end_date, num_high_periods, )
    num_finite_target_level = where_finite_target.size
    where_full_low_periods = _detect_full_low_periods(target_level_data, num_within, )

    high_data = []

    for low_data_variant in self.iter_own_data_variants_from_to(..., ):

        low_data_variant[where_full_low_periods] = _np.nan
        where_finite, *_ = _np.nonzero(_np.isfinite(low_data_variant, ))
        num_finite = where_finite.size

        rho = form.get_rho(low_freq, high_freq, low_data_variant, )
        const = form.get_constant(low_freq, high_freq, low_data_variant, )
        sigma_vector = form.get_sigma_vector(rho, num_high_periods, )

        K = _np.zeros((num_high_periods-1, num_high_periods), dtype=float, )
        C = _np.full((num_high_periods-1, 1), const, dtype=float)
        for i in range(num_high_periods-1, ):
            K[i, i+1] = 1 / sigma_vector[i+1]
            K[i, i] = -rho / sigma_vector[i+1]
            C[i, 0] = const / sigma_vector[i+1]
        F = K.T @ K
        C = K.T @ C

        multiplier_columns = tuple(
            create_multiplier_column(i, )
            for i in where_finite
        )

        target_level_columns = tuple(
            create_target_level_column(i, )
            for i in where_finite_target
        )

        right_padding = _np.zeros((1, num_finite+num_finite_target_level, ), dtype=float, )

        aggregation_rows = tuple(
            _np.hstack((create_aggregation_row(i, ), right_padding, ), )
            for i in where_finite
        )

        target_level_rows = tuple(
            _np.hstack((create_target_level_row(i, ), right_padding, ), )
            for i in where_finite_target
        )

        F = _np.hstack((F, *multiplier_columns, *target_level_columns, ))
        F = _np.vstack((F, *aggregation_rows, *target_level_rows, ))

        C = _np.vstack((
            C,
            low_data_variant[where_finite].reshape(-1, 1, ),
            target_level_data[where_finite_target].reshape(-1, 1, ),
        ))

        high_data_variant = _np.linalg.solve(F, C, )
        high_data_variant = high_data_variant[:num_high_periods]
        high_data.append(high_data_variant, )

    high_data = _np.hstack(high_data, )
    return high_start_date, high_data
    #]


class RateForm:
    """
    """
    #[

    @staticmethod
    def get_rho(low_freq, high_freq, low_data_variant, ) -> Real:
        """
        """
        where_finite, *_ = _np.nonzero(_np.isfinite(low_data_variant, ))
        num_low_roc_periods = where_finite[-1] - where_finite[0]
        low_roc = (
            (low_data_variant[where_finite[-1]] / low_data_variant[where_finite[0]])
            ** (1 / num_low_roc_periods)
        )
        return _conversions.convert_roc(low_roc, low_freq, high_freq, )

    @staticmethod
    def get_constant(*args, ) -> Real:
        return 0

    @staticmethod
    def get_sigma_vector(rho, num_high_periods, ) -> _np.ndarray:
        return rho ** _np.arange(num_high_periods, )

    #]


class DiffForm:
    """
    """
    #[

    @staticmethod
    def get_rho(low_freq, high_freq, low_data_variant, ) -> Real:
        """
        """
        return 1

    def get_constant(low_freq, high_freq, low_data_variant, ) -> Real:
        """
        """
        where_finite, *_ = _np.nonzero(_np.isfinite(low_data_variant, ))
        num_low_diff_periods = where_finite[-1] - where_finite[0]
        low_diff = (
            (low_data_variant[where_finite[-1]] - low_data_variant[where_finite[0]])
            / num_low_diff_periods
        )
        return _conversions.convert_diff(low_diff, low_freq, high_freq, )

def _create_aggregation_vector_sum(num_within, ):
    return [1, ] * num_within


def _create_aggregation_vector_mean(num_within, ):
    return [1/num_within, ] * num_within


def _create_aggregation_vector_first(num_within, ):
    return [1, ] + [0, ]*(num_within-1)


def _create_aggregation_vector_last(num_within, ):
    return [0, ]*(num_within-1) + [1, ]


_CHOOSE_FORM = {
    "rate": RateForm,
    "diff": DiffForm,
}


_CHOOSE_AGGREGATION = {
    "sum": _create_aggregation_vector_sum,
    "mean": _create_aggregation_vector_mean,
    "avg": _create_aggregation_vector_mean,
    "first": _create_aggregation_vector_first,
    "last": _create_aggregation_vector_last,
}


def _create_multiplier_column(low_period, num_low_periods, num_within, ):
    multiplier_column = _np.zeros((num_low_periods, num_within, ), dtype=float, )
    multiplier_column[low_period, :] = 1
    return multiplier_column.reshape(-1, 1)


def _create_target_level_column(high_period, num_high_periods, ):
    target_level_column = _np.zeros((num_high_periods, ), dtype=float, )
    target_level_column[high_period] = 1
    return target_level_column.reshape(-1, 1, )


def _create_aggregation_row(low_period, num_low_periods, num_within, aggregation_vector, ):
    out = _np.zeros((num_low_periods, num_within, ), dtype=float, )
    out[low_period, :] = aggregation_vector
    return out.reshape(1, -1, )


def _create_target_level_row(high_period, num_high_periods, ):
    target_level_row = _np.zeros((num_high_periods, ), dtype=float, )
    target_level_row[high_period] = 1
    return target_level_row.reshape(1, -1, )


def _detect_full_low_periods(target_level_data, num_within, ):
    x = target_level_data.reshape((-1, num_within, ))
    return _np.where(_np.all(_np.isfinite(x, ), axis=1, ))


def _get_target_info(
    target_level,
    high_start_date,
    high_end_date,
    num_high_periods,
    /,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    """
    target_level_data = (
        target_level.get_data_from_to((high_start_date, high_end_date), )
        if target_level is not None else _np.full((num_high_periods, ), _np.nan, dtype=float, )
    )
    where_finite_target, *_ = _np.nonzero(_np.isfinite(target_level_data, ))
    return target_level_data, where_finite_target

