"""
Univariate time series filters
"""


#[
from __future__ import annotations

from numbers import (Number, )
from collections.abc import (Iterable, Callable, )
from types import (EllipsisType, )
from typing import (Self, )
import numpy as _np

from .. import dates as _dates
#]


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
        smooth: Number,
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
        I = _np.eye(self._num_periods, dtype=float)
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
        extra_columns = _np.zeros((self._num_periods + num_constraints, num_constraints, ), dtype=float)
        for i, j in enumerate(level_where, ):
            extra_rows[i, j] = 1
            extra_columns[j, i] = 1
        self._F = _np.vstack((self._F, extra_rows, ))
        self._F = _np.hstack((self._F, extra_columns, ))

    def _add_change_constraints(self, change_where: list[int], /, ):
        if not change_where:
            return
        num_constraints = len(change_where)
        extra_rows = _np.zeros((num_constraints, self._F.shape[1], ), dtype=float, )
        extra_columns = _np.zeros((self._F.shape[0] + num_constraints, num_constraints, ), dtype=float, )
        for i, j in enumerate(change_where, ):
            extra_rows[i, [j-1, j]] = (-1, 1)
            extra_columns[[j-1, j], i] = (-1, 1)
        self._F = _np.vstack((self._F, extra_rows, ))
        self._F = _np.hstack((self._F, extra_columns, ))
        self._num_extra_rows += num_constraints
    #]


class Mixin:
    """
    Constrained Hodrick-Prescott filter mixin for Series
    """
    #[
    def hpf(
        self,
        *,
        range: Iterable[_dates.Dater] | EllipsisType = ...,
        smooth: Number | None = None,
        log: bool = False,
        level: Self | None = None,
        change: Self | None = None,
    ) -> tuple[Self, Self, ]:
        """
        Constrained Hodrick-Prescott filter
        """
        if smooth is None:
            smooth = _get_default_smooth(self.frequency)
        #
        range = self._resolve_dates(range)
        encompassing_range = _dates.get_encompassing_range(self, level, change, range, )
        range = [ t for t in encompassing_range ]
        data = self.get_data(range)
        num_periods, num_columns = data.shape
        level_data, level_where = _prepare_constraints(level, range, )
        change_data, change_where = _prepare_constraints(change, range, )
        change_data, change_where = _remove_first_date_change(change_data, change_where, )
        #
        hp = _ConstrainedHodrickPrescottFilter(
            num_periods,
            smooth,
            level_where=level_where,
            change_where=change_where,
            log=log,
        )
        trend_data, gap_data = hp.filter_data(data, level_data=level_data, change_data=change_data, )
        #
        trend = type(self).from_dates_and_values(range, trend_data, )
        gap = type(self).from_dates_and_values(range, gap_data, )
        return trend, gap

    def hpf_trend(self, /, *args, **kwargs):
        return self.hpf(*args, **kwargs, )[0]

    def hpf_gap(self, /, *args, **kwargs):
        return self.hpf(*args, **kwargs, )[1]
    #]


def _prepare_constraints(
    constraint: Self | None,
    range: list[_dates.Dater],
    /,
) -> tuple[_np.ndarray | None, list[int] | None, ]:
    #[
    if constraint is None:
        return None, None
    data = constraint.get_data(range, 0)
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

