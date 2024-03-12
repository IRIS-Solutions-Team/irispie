"""
"""

#[
from __future__ import annotations

import dataclasses as _dc
import numpy as _np

from .. import dates as _dates
from ..dataslates import main as _dataslates
from ..databoxes import main as _databoxes

from . import solutions as _solutions
from . import initializers as _initializers
#]


@_dc.dataclass
class HumanKalmanOutputData:
    """
    """
    #[

    predict_mean = None
    predict_std = None

    update_mean = None
    update_std = None
    predict_error = None

    smooth_mean = None
    smooth_std = None

    predict_mse_measurement = None

    @classmethod
    def from_kalman_output_store(
        klass,
        kos: _KalmanOutputStore,
    ) -> Self:
        """
        """
        self = klass()
        self.predict_mse_measurement = kos.predict_mse_measurement
        if kos.needs.return_predict:
            self.predict_mean = kos.predict_mean.to_databox()
            self.predict_std = kos.predict_std.to_databox()
        if kos.needs.return_update:
            self.update_mean = kos.update_mean.to_databox()
            self.update_std = kos.update_std.to_databox()
            self.predict_error = kos.predict_error.to_databox()
        if kos.needs.return_smooth:
            self.smooth_mean = kos.smooth_mean.to_databox()
            self.smooth_std = kos.smooth_std.to_databox()
        return self

    def __repr__(self, /, ) -> str:
        """
        """
        return f"HumanKalmanOutputData({', '.join(self.__dict__.keys())})"


    #]


class _KalmanNeeds:
    """
    """
    #[

    return_predict = True
    return_update = True
    return_smooth = True
    rescale_variance = False

    def __init__(self: Self, **kwargs, ) -> None:
        """
        """
        for k, v in kwargs.items():
            if not hasattr(self, k) or getattr(self, k) is None:
                continue
            setattr(self, k, v)

    #]


@_dc.dataclass
class _KalmanOutputStore:
    """
    """
    #[

    needs = None

    predict_mean = None
    predict_std = None

    update_mean = None
    update_std = None
    predict_error = None

    smooth_mean = None
    smooth_std = None

    predict_mse_measurement = None

    neg_log_lik = None
    var_scale = None
    diffuse_factor = None

    def __init__(
        self,
        ds: _dataslates.Dataslate,
        needs: _KalmanNeeds,
        y_names: Iterable[str],
    ) -> Self:
        """
        """
        self.needs = needs
        self.neg_log_lik = [_np.full((1, ), _np.nan, dtype=_np.float64, )]
        self.var_scale = [_np.full((1, ), _np.nan, dtype=_np.float64, )]
        self.diffuse_factor = [_np.full((1, ), _np.nan, dtype=_np.float64, )]
        if self.needs.return_predict:
            self.predict_mean = ds.nan_copy()
            self.predict_std = ds.nan_copy()
        if self.needs.return_update:
            self.update_mean = ds.nan_copy()
            self.update_std = ds.nan_copy()
            self.predict_error = _dataslates.Dataslate.nan_from_names_dates(
                y_names, ds.dates,
                base_columns=ds.base_columns,
            )
        if self.needs.return_smooth:
            self.smooth_mean = ds.nan_copy()
            self.smooth_std = ds.nan_copy()

    def create_output_args(self, /, ) -> tuple[HumanKalmanOutputData, dict[str, Any]]:
        """
        """
        return HumanKalmanOutputData.from_kalman_output_store(self, ), self._create_info()

    def _create_info(self, /, ) -> dict[str, Any]:
        return {
            "neg_log_lik": [ float(i[0]) for i in self.neg_log_lik ],
            "var_scale": [ float(i[0]) for i in self.var_scale ],
            "diffuse_factor": [ float(i[0]) for i in self.diffuse_factor ],
        }

    #]


@_dc.dataclass
class _KalmanRunningVariant:
    """
    """
    #[
    LOG_2_PI = _np.log(2 * _np.pi)

    base_columns = None
    nonbase_columns = None
    needs = None

    all_num_obs = None
    all_pex_invFx_pex = None
    all_invFx_pex = None
    all_det_Fx = None

    all_L = None
    all_a0 = None
    all_P0 = None
    all_ZaxT_invFx_Zax = None
    all_ZaxT_invFx_pex = None
    all_G = None
    all_inx_y = None

    sum_num_obs = None
    sum_log_det_Fx = None
    sum_pex_invFx_pex = None

    predict_mean_array = None
    predict_std_array = None
    update_mean_array = None
    update_std_array = None
    predict_error_data = None
    smooth_mean_array = None
    smooth_std_array = None

    y1_array = None
    u0_array = None
    v0_array = None
    w0_array = None

    neg_log_lik = None
    var_scale = None
    diffuse_factor = None

    def __init__(
        self: Self,
        ds: _dataslates.Dataslate,
        needs: _KalmanNeeds,
    ) -> None:
        """
        """
        num_ext_periods = ds.num_periods
        self.base_columns = ds.base_columns
        self.nonbase_columns = ds.nonbase_columns
        self.needs = needs
        #
        self.all_num_obs = [None] * num_ext_periods
        self.all_pex_invFx_pex = [None] * num_ext_periods
        self.all_invFx_pex = [None] * num_ext_periods
        self.all_det_Fx = [None] * num_ext_periods
        self.all_L = [None] * num_ext_periods
        self.all_a0 = [None] * num_ext_periods
        self.all_P0 = [None] * num_ext_periods
        self.all_ZaxT_invFx_Zax = [None] * num_ext_periods
        self.all_ZaxT_invFx_pex = [None] * num_ext_periods
        self.all_G = [None] * num_ext_periods

    def calculate_likelihood( self, /, ) -> None:
        """
        """
        self.sum_num_obs = sum(i for i in self.all_num_obs if i is not None)
        self.sum_log_det_Fx = sum(i for i in self.all_det_Fx if i is not None)
        self.sum_pex_invFx_pex = sum(i for i in self.all_pex_invFx_pex if i is not None)
        #
        _calculate_variance_scale(self, )
        #
        self.neg_log_lik[0] = (
            + self.sum_num_obs*self.LOG_2_PI
            + self.sum_log_det_Fx
            + self.sum_pex_invFx_pex
        ) / 2;

    def rescale_stds(self, /, ) -> None:
        """
        """
        if not self.needs.rescale_variance:
            return
        std_scale = _sqrt_positive(self.var_scale)
        if self.needs.return_predict:
            self.predict_std_array *= _sqrt_positive(self.var_scale, )
        if self.needs.return_update:
            self.update_std_array *= _sqrt_positive(self.var_scale, )
        if self.needs.return_smooth:
            self.smooth_std_array *= _sqrt_positive(self.var_scale, )

    #]


right_div = _solutions.right_div
left_div = _solutions.left_div


def _sqrt_positive(x):
    return _np.sqrt(_np.maximum(x, 0.0, ))


def _std_from_mse(mse, ):
    return _sqrt_positive(_np.diag(mse).copy())


def _store_std_from_mse(std_array, array_indexes, mse, pos=None, ) -> None:
    if pos is not None:
        mse = mse[pos, :][:, pos]
    std_array[array_indexes] = _std_from_mse(mse, )


def _store_mean(mean_array, array_indexes, mean, pos=None, ) -> None:
    if pos is not None:
        mean = mean[pos]
    mean_array[array_indexes] = mean


def _calculate_variance_scale(krv: _KalmanRunningVariant, /, ) -> None:
    """
    """
    #[
    if not krv.needs.rescale_variance:
        krv.var_scale[0] = 1
        return
    if krv.sum_num_obs > 0:
        krv.var_scale[0] = krv.sum_pex_invFx_pex / krv.sum_num_obs
        krv.sum_log_det_Fx += krv.sum_num_obs * _np.log(krv.var_scale[0])
        krv.sum_pex_invFx_pex = krv.sum_pex_invFx_pex / krv.var_scale[0]
    else:
        krv.var_scale[0] = 1
        krv.sum_pex_invFx_pex = 0
    #]


class KalmanMixin:
    """
    """
    #[

    def kalman_filter(
        self,
        #
        input_db: _databoxes.Databox,
        span: Iterable[_dates.Dater],
        #
        diffuse_factor: Real | None = None,
        #
        return_predict: bool = True,
        return_update: bool = True,
        return_smooth: bool = True,
        rescale_variance: bool = False,
    ) -> _databoxes.Databox:
        """
        """
        #[

        ds = _dataslates.Dataslate.from_databox_for_slatable(
            self,
            input_db,
            span,
        )

        needs = _KalmanNeeds(
            return_predict=return_predict,
            return_update=return_update,
            return_smooth=return_smooth,
            rescale_variance=rescale_variance,
        )

        qid_to_name = self.create_qid_to_name()
        qid_to_logly = self.create_qid_to_logly()
        solution_vectors = self._solution_vectors
        temp = [(t.qid, i) for i, t in enumerate(solution_vectors.transition_variables) if not t.shift ]
        curr_qids, curr_pos = tuple(zip(*temp))
        curr_qids = list(curr_qids)
        curr_pos = list(curr_pos)

        y_qids = [ t.qid for t in solution_vectors.measurement_variables ]
        u_qids = [ t.qid for t in solution_vectors.unanticipated_shocks ]
        v_qids = [ t.qid for t in solution_vectors.anticipated_shocks ]
        w_qids = [ t.qid for t in solution_vectors.measurement_shocks ]

        y_names = [ qid_to_name[qid] for qid in y_qids ]
        u_names = [ qid_to_name[qid] for qid in u_qids ]
        v_names = [ qid_to_name[qid] for qid in v_qids ]
        w_names = [ qid_to_name[qid] for qid in w_qids ]


        kos = _KalmanOutputStore(ds, needs, y_names, )


        #
        # Variant dependent from here
        #
        dataslate_v = ds.get_variant(0)

        krv = _KalmanRunningVariant(dataslate_v, needs, )

        solution = self.get_solution_matrices()
        cov_u = self.get_cov_unanticipated_shocks()
        cov_w = self.get_cov_measurement_shocks()

        #
        # Initialize mean and MSE
        #
        init_mean, init_mse, diffuse_factor_resolved = _initializers.initialize(
            solution, cov_u,
            diffuse_scale=diffuse_factor,
        )

        krv.diffuse_factor = kos.diffuse_factor[0]
        krv.diffuse_factor[0] = diffuse_factor_resolved

        a1, P1 = init_mean, init_mse
        Ta = solution.Ta
        Ra = solution.Ra
        Pa = solution.Pa
        Ka = solution.Ka
        Za = solution.Za
        H = solution.H
        D = solution.D
        Ua = solution.Ua

        tolerance = 1e-12
        Za[_np.abs(Za)<tolerance] = 0
        Ua[_np.abs(Ua)<tolerance] = 0
        Pa[_np.abs(Pa)<tolerance] = 0

        num_a = Ta.shape[0]
        num_u = Pa.shape[1]
        num_v = Ra.shape[1]
        num_y = Za.shape[0]
        num_w = H.shape[1]



        data_array = dataslate_v.get_data_variant(0, )
        data_array[_np.ix_(y_qids, krv.nonbase_columns)] = _np.nan
        data_array[_np.ix_(u_qids, krv.nonbase_columns)] = _np.nan
        data_array[_np.ix_(v_qids, krv.nonbase_columns)] = _np.nan
        data_array[_np.ix_(w_qids, krv.nonbase_columns)] = _np.nan


        krv.y1_array = data_array[y_qids, :]
        krv.all_inx_y = tuple(
            _np.isfinite(column_data, )
            for column_data in krv.y1_array.T
        )

        num_ext_periods = ds.num_periods
        krv.u0_array = _np.zeros((num_u, num_ext_periods), dtype=_np.float64, )
        _np.nan_to_num(krv.u0_array, nan=0.0, copy=False, )
        krv.u0_array[:, krv.nonbase_columns] = _np.nan

        krv.v0_array = _np.zeros((num_v, num_ext_periods), dtype=_np.float64, )
        _np.nan_to_num(krv.v0_array, nan=0.0, copy=False, )
        krv.v0_array[:, krv.nonbase_columns] = _np.nan

        krv.w0_array = _np.zeros((num_w, num_ext_periods), dtype=_np.float64, )
        _np.nan_to_num(krv.w0_array, nan=0.0, copy=False, )
        krv.w0_array[:, krv.nonbase_columns] = _np.nan


        if needs.return_predict:
            krv.predict_mean_array = kos.predict_mean.get_data_variant(0, )
            krv.predict_mean_array[u_qids, :] = krv.u0_array
            krv.predict_mean_array[v_qids, :] = krv.v0_array
            krv.predict_mean_array[w_qids, :] = krv.w0_array
            krv.predict_std_array = kos.predict_std.get_data_variant(0, )

        if needs.return_update:
            krv.update_mean_array = kos.update_mean.get_data_variant(0, )
            krv.update_std_array = kos.update_std.get_data_variant(0, )
            krv.predict_error_data = kos.predict_error.get_data_variant(0, )

        if needs.return_smooth:
            krv.smooth_mean_array = kos.smooth_mean.get_data_variant(0, )
            krv.smooth_std_array = kos.smooth_std.get_data_variant(0, )

        krv.neg_log_lik = kos.neg_log_lik[0]
        krv.var_scale = kos.var_scale[0]
        krv.diffuse_factor = kos.diffuse_factor[0]


        for t in krv.base_columns:

            inx_y = krv.all_inx_y[t]

            # MSE prediction step
            P0 = Ta @ P1 @ Ta.T + Pa @ cov_u @ Pa.T
            P0 = (P0 + P0.T) / 2
            F = Za @ P0 @ Za.T + H @ cov_w @ H.T
            F = (F + F.T) / 2
            if needs.return_predict:
                _store_std_from_mse(krv.predict_std_array, (curr_qids, t), Ua @ P0 @ Ua.T, curr_pos, )

            # Extract time-varying means for shocks
            u0 = krv.u0_array[:, t]
            v0 = krv.v0_array[:, t]
            w0 = krv.w0_array[:, t]

            # Mean prediction step
            a0 = Ta @ a1 + Ka + Pa @ u0
            y0 = Za @ a0 + D + H @ w0
            y0[~inx_y] = _np.nan

            if needs.return_predict:
                _store_mean(krv.predict_mean_array, (curr_qids, t), Ua @ a0, curr_pos, )
                _store_mean(krv.predict_mean_array, (y_qids, t), y0, )
                _store_mean(krv.predict_mean_array, (u_qids, t), u0, )
                _store_mean(krv.predict_mean_array, (v_qids, t), v0, )
                _store_mean(krv.predict_mean_array, (w_qids, t), w0, )

            # MSE updating step
            Fx = F[inx_y, :][:, inx_y]
            Zax = Za[inx_y, :]
            ZaxT_invFx = right_div(Zax.T, Fx, )
            G = P0 @ ZaxT_invFx
            P1 = P0 - G @ Zax @ P0
            P1 = (P1 + P1.T) / 2

            if needs.return_update:
                _store_std_from_mse(krv.update_std_array, (curr_qids, t), Ua @ P1 @ Ua.T, curr_pos, )

            # Mean updating step
            krv.all_num_obs[t] = _np.count_nonzero(inx_y, )
            y1 = krv.y1_array[:, t]
            pe = y1 - y0
            pex = pe[inx_y]
            a1 = a0 + G @ pex

            invFx_pex = left_div(Fx, pex, ) # inv(Fx) * pex

            if needs.return_update:
                Hx_cov_w = H[inx_y, :] @ cov_w
                w1 = w0 + Hx_cov_w.T @ invFx_pex
                Pa_cov_u = Pa @ cov_u
                r = ZaxT_invFx @ pex # Z' * inv(F) * pe
                u1 = u0 + Pa_cov_u.T @ r
                v1 = v0
                _store_mean(krv.update_mean_array, (curr_qids, t), Ua @ a1, curr_pos, )
                _store_mean(krv.update_mean_array, (y_qids, t), y1, )
                _store_mean(krv.update_mean_array, (u_qids, t), u1, )
                _store_mean(krv.update_mean_array, (v_qids, t), v1, )
                _store_mean(krv.update_mean_array, (w_qids, t), w1, )
                _store_mean(krv.predict_error_data, (..., t), pe, )

            if needs.return_smooth:
                krv.all_a0[t] = a0
                krv.all_P0[t] = P0
                krv.all_ZaxT_invFx_pex[t] = Zax.T @ invFx_pex
                krv.all_ZaxT_invFx_Zax[t] = ZaxT_invFx @ Zax
                krv.all_L[t] = Ta - (Ta @ G) @ Zax
                krv.all_G[t] = G
                krv.all_invFx_pex[t] = invFx_pex

            krv.all_pex_invFx_pex[t] = pex @ invFx_pex
            krv.all_det_Fx[t] = _np.log(_np.linalg.det(Fx))


        reversed_base_columns = (
            reversed(krv.base_columns, )
            if needs.return_smooth else ()
        )

        N = None
        r = None

        for t in reversed_base_columns:

            inx_y = krv.all_inx_y[t]
            u0 = krv.u0_array[:, t]
            v0 = krv.v0_array[:, t]
            w0 = krv.w0_array[:, t]

            a0 = krv.all_a0[t]
            L = krv.all_L[t]
            G = krv.all_G[t]
            P0 = krv.all_P0[t]
            ZaxT_invFx_pex = krv.all_ZaxT_invFx_pex[t]
            ZaxT_invFx_Zax = krv.all_ZaxT_invFx_Zax[t]
            invFx_pex = krv.all_invFx_pex[t]

            N = ZaxT_invFx_Zax + ((L.T @ N @ L) if N is not None else 0)
            P2 = P0 - P0 @ N @ P0
            _store_std_from_mse(krv.smooth_std_array, (curr_qids, t), Ua @ P2 @ Ua.T, curr_pos, )

            H_cov_w = H[inx_y, :] @ cov_w
            w2 = w0 + H_cov_w.T @ (invFx_pex - (((Ta @ G).T @ r) if r is not None else 0))
            _store_mean(krv.smooth_mean_array, (w_qids, t), w2, )

            r = ZaxT_invFx_pex + ((L.T @ r) if r is not None else 0)
            a2 = a0 + P0 @ r
            _store_mean(krv.smooth_mean_array, (curr_qids, t), Ua @ a2, curr_pos, )

            y2 = krv.y1_array[:, t]
            _store_mean(krv.smooth_mean_array, (y_qids, t), y2, )

            Pa_cov_u = Pa @ cov_u
            u2 = u0 + Pa_cov_u.T @ r
            _store_mean(krv.smooth_mean_array, (u_qids, t), u2, )

            v2 = v0
            _store_mean(krv.smooth_mean_array, (v_qids, t), v2, )

        krv.calculate_likelihood()
        krv.rescale_stds()

        return kos.create_output_args()

    #]


