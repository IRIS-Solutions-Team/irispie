"""
Univariate time series filters
"""


#[
from __future__ import annotations

from numbers import (Real, )
from collections.abc import (Iterable, Callable, )
from types import (EllipsisType, )
from typing import (Self, )
import numpy as _np

from .. import dates as _dates
from . import main as _series
from . import _functionalize
#]


__all__ = ("hpf", )


_AUTO_SMOOTH = {
    _dates.Frequency.YEARLY: (10*1)**2,
    _dates.Frequency.HALFYEARLY: (10*2)**2,
    _dates.Frequency.QUARTERLY: (10*4)**2,
    _dates.Frequency.MONTHLY: (10*12)**2,
    "default": (10*4)**2,
}


def _get_default_smooth(frequency, /, ):
    """
    Default smoothing parameter (lambda)
    """
    return _AUTO_SMOOTH.get(frequency, _AUTO_SMOOTH["default"])


class _ConstrainedHodrickPrescottFilter:
    """
    """
    #[
    def __init__(
        self,
        num_periods: int,
        smooth: Real,
        /,
        level_where: list[int] | None = None,
        change_where: list[int] | None = None,
        log: bool = False,
    ) -> None:
        self._num_periods = num_periods
        self._smooth = smooth
        self._log = log
        self._num_extra_rows = 0
        self._create_plain_filter_matrix()
        self._add_level_constraints(level_where, )
        self._add_change_constraints(change_where, )

    def filter_data(
        self,
        data: _np.ndarray,
        /,
        level_data: _np.ndarray | None = None,
        change_data: _np.ndarray | None = None,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        """
        data = data.reshape(-1, 1)
        F = self._add_eye_for_observations(data, )
        extended_data = self._extend_data(data, level_data, change_data, )
        if self._log:
            extended_data = _np.log(extended_data, )
        enforced_zeros_where = _np.where(_np.isnan(data, ))
        extended_data[enforced_zeros_where] = 0
        trend_data = _np.linalg.solve(F, extended_data, )
        extended_data[enforced_zeros_where] = _np.nan
        gap_data = extended_data - trend_data
        if self._num_extra_rows > 0:
            trend_data = trend_data[:-self._num_extra_rows, :]
            gap_data = gap_data[:-self._num_extra_rows, :]
        if self._log:
            trend_data = _np.exp(trend_data, )
            gap_data = _np.exp(gap_data, )
        return trend_data, gap_data

    def _add_eye_for_observations(
        self,
        data: _np.ndarray,
        /,
    ) -> None:
        """
        Add 1 to the F diagonal for each row where an observation is available
        """
        #[
        F = _np.copy(self._F)
        quasi_eye = _np.diag(_np.float64(~_np.isnan(data.flatten(), )))
        F[:self._num_periods, :self._num_periods] += quasi_eye
        return F
        #]

    def _extend_data(
        self,
        data: _np.ndarray,
        level_data: _np.ndarray | None,
        change_data: _np.ndarray | None,
    ) -> _np.ndarray:
        """
        """
        extended_data = data.copy() if data.ndim > 1 else data.copy().reshape(-1, 1)
        tile_dim = (1, extended_data.shape[1], )
        if level_data is not None:
            extended_data = _np.vstack((extended_data, _np.tile(level_data, tile_dim), ), )
        if change_data is not None:
            extended_data = _np.vstack((extended_data, _np.tile(change_data, tile_dim), ), )
        return extended_data

    def _create_plain_filter_matrix(self, ) -> None:
        """
        """
        K = _np.zeros((self._num_periods-2, self._num_periods), dtype=float)
        for i in range(self._num_periods-2):
            K[i,i] = 1
            K[i,i+2] = 1
        for i in range(self._num_periods-2):
            K[i,i+1] = -2
        self._F = self._smooth * (K.T @ K)

    def _add_level_constraints(self, level_where: list[int], /, ):
        if not level_where:
            return
        num_constraints = len(level_where)
        self._num_extra_rows += num_constraints
        extra_rows = _np.zeros((num_constraints, self._num_periods, ), dtype=float)
        extra_variants = _np.zeros((self._num_periods + num_constraints, num_constraints, ), dtype=float)
        for i, j in enumerate(level_where, ):
            extra_rows[i, j] = 1
            extra_variants[j, i] = 1
        self._F = _np.vstack((self._F, extra_rows, ))
        self._F = _np.hstack((self._F, extra_variants, ))

    def _add_change_constraints(self, change_where: list[int], /, ):
        if not change_where:
            return
        num_constraints = len(change_where)
        extra_rows = _np.zeros((num_constraints, self._F.shape[1], ), dtype=float, )
        extra_variants = _np.zeros((self._F.shape[0] + num_constraints, num_constraints, ), dtype=float, )
        for i, j in enumerate(change_where, ):
            extra_rows[i, [j-1, j]] = (-1, 1)
            extra_variants[[j-1, j], i] = (-1, 1)
        self._F = _np.vstack((self._F, extra_rows, ))
        self._F = _np.hstack((self._F, extra_variants, ))
        self._num_extra_rows += num_constraints
    #]


class Inlay:
    """
    Constrained Hodrick-Prescott filter mixin for Series
    """
    #[

    def hpf_trend(self, /, *args, **kwargs):
        start_date, trend_data, _ = _data_hpf(self, *args, **kwargs, )
        self._replace_start_date_and_values(start_date, trend_data, )

    def hpf_gap(self, /, *args, **kwargs):
        start_date, _, gap_data = _data_hpf(self, *args, **kwargs, )
        self._replace_start_date_and_values(start_date, gap_data, )

    #]


def _data_hpf(
    self,
    *,
    span: Iterable[_dates.Dater] | EllipsisType = ...,
    smooth: Real | None = None,
    log: bool = False,
    level: _series.Series | None = None,
    change: _series.Series | None = None,
) -> tuple[_dates.Dater, _np.ndarray, _np.ndarray]:
    """
    Hodrick-Prescott filter run on a multi-variant data matrix
    """
    #[
    if smooth is None:
        smooth = _get_default_smooth(self.frequency, )
    span = self._resolve_dates(span, )
    encompassing_span, *from_to = _dates.get_encompassing_span(self, level, change, span, )
    num_periods = len(encompassing_span, )
    level_data, level_where = _prepare_constraints(level, from_to, )
    change_data, change_where = _prepare_constraints(change, from_to, )
    change_data, change_where = _remove_first_date_change(change_data, change_where, )
    #
    hp = _ConstrainedHodrickPrescottFilter(
        num_periods,
        smooth,
        level_where=level_where,
        change_where=change_where,
        log=log,
    )
    #
    trend_data = []
    gap_data = []
    for data_variant in self.iter_own_data_variants_from_to(from_to, ):
        trend_data_variant, gap_data_variant = hp.filter_data(
            data_variant,
            level_data=level_data,
            change_data=change_data,
        )
        trend_data.append(trend_data_variant, )
        gap_data.append(gap_data_variant, )
    #
    trend_data = _np.hstack(trend_data, )
    gap_data = _np.hstack(gap_data, )
    #
    if span:
        new_start_date = min(span, )
        new_end_date = max(span, )
        clip_start = new_start_date - from_to[0]
        clip_end = new_end_date - from_to[0] + 1
        trend_data = trend_data[clip_start:clip_end, ...]
        gap_data = gap_data[clip_start:clip_end, ...]
    else:
        new_start_date = None
        trend_data = trend_data[[], ...]
        gap_data = gap_data[[], ...]
    #
    return new_start_date, trend_data, gap_data
    #]


def hpf(self, *args, **kwargs, ) -> tuple[_series.Series, _series.Series]:
    """
    Constrained Hodrick-Prescott filter
    """
    start_date, trend_data, gap_data = _data_hpf(self, *args, **kwargs, )
    trend = type(self)(start_date=start_date, values=trend_data, )
    gap = type(self)(start_date=start_date, values=gap_data, )
    return trend, gap


for n in ("hpf_trend", "hpf_gap", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )



def _prepare_constraints(
    constraint: Self | None,
    from_to: tuple[_dates.Dater, _dates.Dater, ],
    /,
) -> tuple[_np.ndarray | None, list[int] | None, ]:
    #[
    if constraint is None:
        return None, None
    data = constraint.get_data_from_to(from_to, 0, )
    where = list(_np.where(~_np.isnan(data.reshape(-1, ), ), )[0])
    data = data[where, :]
    return data, where
    #]


def _remove_first_date_change(
    change_data: _np.ndarray | None,
    change_where: list[int] | None,
) -> tuple[_np.ndarray, list[int], ]:
    #[
    if change_where and 0 in change_where:
        where = change_where.index(0)
        change_where.pop(where)
        change_data = _np.delete(change_data, where, axis=0)
        if not change_where:
            change_data = None
            change_where = None
    return change_data, change_where
    #]

