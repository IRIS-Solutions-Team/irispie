"""
Univariate time series filters
"""


#[
from __future__ import (annotations, )
# from IPython import embed

from numbers import (Number, )
from collections.abc import (Iterable, Callable, )
from types import (EllipsisType, )
import numpy as np_

from ..dataman import (dates as da_, series as se_)
#]


_LAMBDA_FREQUENCIES = [
    da_.Frequency.YEARLY,
    da_.Frequency.HALFYEARLY,
    da_.Frequency.QUARTERLY,
    da_.Frequency.MONTHLY,
]


def _get_default_smooth(frequency, /, ):
    """
    Default smoothing parameter (lambda)
    """
    if frequency not in _LAMBDA_FREQUENCIES:
        raise Exception("Default smoothing parameter (lambda) exists only for yearly, half-yearly, quarterly and monthly frequencies")
    return (10*frequency.value)**2


def _get_filter_matrices(num_periods, /, ):
    I = np_.eye(num_periods, dtype=float)
    K = np_.zeros((num_periods-2, num_periods), dtype=float)
    for i in range(num_periods-2):
        K[i,i] = 1
        K[i,i+2] = 1
    for i in range(num_periods-2):
        K[i,i+1] = -2
    return I, K

class HodrickPrescottMixin:
    """
    Hodrick-Prescott filter mixin for Series
    """
    #[
    def hpf(
        self,
        /,
        range: Iterable[Dater] | EllipsisType = ...,
        smooth: Number | None = None,
        log: bool = False
    ) -> Self:
        if smooth is None:
            smooth = _get_default_smooth(self.frequency)
        range = self._resolve_dates(range)
        range = [ t for t in range ]
        data = self.get_data(range)
        num_periods, num_columns = data.shape
        if log:
            data = np_.log(data)
        I, K = _get_filter_matrices(num_periods)
        trend_data = np_.linalg.solve(I + smooth*(K.T @ K), data)
        gap_data = data - trend_data
        if log:
            trend_data = np_.exp(trend_data)
            gap_data = np_.exp(gap_data)
        trend = se_.Series(num_columns=num_columns)
        trend.set_data(range, trend_data)
        gap = se_.Series(num_columns=num_columns)
        gap.set_data(range, gap_data)
        return trend, gap

    def hpf_trend(self, /, *args, **kwargs):
        return self.hpf(*args, **kwargs, )[0]

    def hpf_gap(self, /, *args, **kwargs):
        return self.hpf(*args, **kwargs, )[1]
    #]

