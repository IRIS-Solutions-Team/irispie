r"""
"""


#[

from __future__ import annotations

import functools as _ft
import numpy as _np

from ..databoxes import Databox
from ..dataslates import Dataslate
from ..dates import Period
from ..fords import least_squares as _least_squares
from .. import dates as _periods

from ._variants import Variant
from . import prior_obs as _prior_obs
from .prior_obs import PriorObs
from ._dimensions import Dimensions
from ..progress_bars import ProgressBar
from ..fords import covariances as _covariances

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Callable, Literal

#]


_GET_SPANS_DISPATCH = {
    "short": _periods.spans_from_short_span,
    "long": _periods.spans_from_long_span,
}


class Inlay:
    r"""
    """
    #[

    def estimate(
        self,
        input_data: Databox,
        span: Iterable[Period] | None = None,
        #
        interpret_span: Literal["short", "long"] = "short",
        num_variants: int | None = None,
        omit_missing: bool = True,
        prior_obs: PriorObs | Iterable[PriorObs, ] | None = None,
        dof_correction: bool = False,
        target_db: Databox | None = None,
        show_progress: bool = False,
        progress_bar_settings: dict = dict(title="Estimating RedVAR", ),
    ) -> Databox:
        r"""
        """
        dimensions = self._invariant.dimensions
        has_intercept = dimensions.has_intercept
        short_span, long_span, = _GET_SPANS_DISPATCH[interpret_span](span, )
        num_variants = self.resolve_num_variants_in_context(num_variants, )
        slatable = self.slatable_for_estimate()
        dataslate = Dataslate.from_databox_for_slatable(
            slatable, input_data, short_span,
            num_variants=num_variants,
        )
        #
        zipped = zip(
            range(num_variants, ),
            dataslate.iter_variants(),
        )
        self._variants = []
        #
        # Estimate each variant
        #=======================================================================
        progress_bar = ProgressBar(
            num_steps=num_variants,
            show_progress=show_progress,
            **progress_bar_settings,
        )
        for vid, dataslate_v in zipped:
            estimated_variant = _estimate_variant(
                self._invariant,
                dataslate_v,
                omit_missing=omit_missing,
                prior_obs=prior_obs,
                dof_correction=dof_correction,
            )
            self._variants.append(estimated_variant, )
            progress_bar.increment()
        #=======================================================================
        #
        output_db = dataslate.to_databox()
        if target_db is not None:
            output_db = target_db | output_db
        return output_db
    #]


def _estimate_variant(
    invariant: Invariant,
    dataslate: Dataslate,
    prior_obs: PriorObs | Iterable[PriorObs] | None,
    dof_correction: bool,
    omit_missing: bool,
) -> Variant:
    r"""
    """
    #[
    dimensions = invariant.dimensions
    order = dimensions.order
    has_intercept = dimensions.has_intercept
    num_endogenous = dimensions.num_endogenous
    num_rhs = dimensions.num_nonendogenous
    num_lagged_endogenous = dimensions.num_lagged_endogenous
    #
    y0, y1, x, k, where, = _get_estimation_data(invariant, dataslate, omit_missing, )
    num_periods_fitted = int(where.sum())
    if num_periods_fitted == 0:
        raise ValueError("No data available for estimation after removing periods with missing observations.")
    #
    lhs_est = y0[:, where]
    rhs_est = _np.vstack([y1, x, k, ])[:, where]
    #
    if prior_obs is not None:
        xk = _np.vstack([x, k, ], )
        gamma = _least_squares.ordinary_least_squares(y0[:, where], xk[:, where], )
        y0_demeaned = y0[:, where] - gamma @ xk[:, where]
        y0_std = _np.sqrt(_np.diag(y0_demeaned @ y0_demeaned.T / num_periods_fitted))
        y0_std = y0_std.reshape((-1, ), )
        lhs_dummy, rhs_dummy, = _prior_obs.arrays_from_prior_obs(prior_obs, dimensions, y0_std, )
        lhs_est = _np.hstack([lhs_est, lhs_dummy, ])
        rhs_est = _np.hstack([rhs_est, rhs_dummy, ])
    #
    beta = _least_squares.ordinary_least_squares(lhs_est, rhs_est, )
    #
    A = beta[:, :num_lagged_endogenous]
    if has_intercept:
        B = beta[:, num_lagged_endogenous:-1]
        c = beta[:, -1]
    else:
        B = beta[:, num_lagged_endogenous:]
        c = None
    #
    u = y0 - A @ y1 - B @ x - c.reshape((-1, 1))
    #
    num_periods_corrected = num_periods_fitted - (num_rhs if dof_correction else 0)
    cov_residuals = u[:, where] @ u[:, where].T / num_periods_corrected
    cov_residuals = _covariances.symmetrize(cov_residuals, )
    #
    fitted_periods = tuple(
        period for period, been_fitted in zip(dataslate.base_periods, where, )
        if been_fitted
    )
    #
    estimated_variant = Variant(
        A=A,
        B=B,
        c=c,
        cov_residuals=cov_residuals,
        fitted_periods=fitted_periods,
        residual_estimates=u,
    )
    #
    _write_residual_estimates(
        invariant,
        dataslate,
        u,
    )
    #
    return estimated_variant
    #]


def _get_estimation_data(
    invariant: Invariant,
    dataslate: Dataslate,
    omit_missing: bool,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    r"""
    """
    #[
    data = dataslate.get_data_variant()
    order = invariant.dimensions.order
    has_intercept = invariant.dimensions.has_intercept
    endogenous_qids = list(invariant.get_endogenous_qids())
    exogenous_qids = list(invariant.get_exogenous_qids())
    y = data[endogenous_qids, :]
    y0 = y[:, order:]
    y1 = _np.vstack([
        y[:, order-i:-i]
        for i in range(1, order+1, )
    ])
    x = data[exogenous_qids, order:]
    k = _np.ones((int(has_intercept), x.shape[1], ), dtype=float, )
    if omit_missing:
        where = get_where_observations([y0, y1, x, k], )
    else:
        num_periods = y0.shape[1]
        where = _np.ones(num_periods, dtype=bool, )
    #
    return y0, y1, x, k, where,
    #]


def _write_residual_estimates(
    invariant: Invariant,
    dataslate: Dataslate,
    residual_estimates: _np.ndarray,
) -> None:
    r"""
    """
    #[
    order = invariant.dimensions.order
    residual_qids = invariant.get_residual_qids()
    data_array = dataslate.get_data_variant()
    data_array[residual_qids, order:] = residual_estimates
    #]


def get_where_observations(data_matrices: list[_np.ndarray], ) -> _np.ndarray:
    r"""
    """
    return _np.all(_np.isfinite(_np.vstack(data_matrices, ), ), axis=0, )


def _add_intercept_dummy(x: _np.ndarray, ) -> _np.ndarray:
    r"""
    """
    return _np.pad(
        x,
        ((0, 1), (0, 0), ),
        mode="constant",
        constant_values=1,
    )

