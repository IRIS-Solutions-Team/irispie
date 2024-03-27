"""
"""


#[
from __future__ import annotations

from typing import (Any, Iterable, )
from numbers import (Real, )
import functools as _ft
import dataclasses as _dc
import numpy as _np
import scipy as _sp

from ..dataslates.main import (Dataslate, )
from ..databoxes.main import (Databox, )
from .solutions import (right_div, left_div, )

from ..dates import (Dater, )
from .. import quantities as _quantities
from . import initializers as _initializers
#]


class KalmanOutputData:
    """
......................................................................

==Output data from Kalman filter==


......................................................................
    """
    #[

    __slots__ = (
        "predict_mse_measurement",
        "predict_med",
        "predict_std",
        "predict_err",
        "update_med",
        "update_std",
        "smooth_med",
        "smooth_std",
    )

    def __init__(
        self: Self,
        output_store: _OutputStore,
    ) -> None:
        """
        """
        for attr_name in self.__slots__:
            attr = getattr(output_store, attr_name, )
            if hasattr(attr, "to_output_arg", ):
                attr = attr.to_output_arg()
            setattr(self, attr_name, attr, )

    def __repr__(self, /, ) -> str:
        """
        """
        slots_to_repr = (s for s in self.__slots__ if getattr(self, s) is not None)
        return f"KalmanOutputData({', '.join(slots_to_repr)})"

    #]


class _Needs:
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


class _LogDataslate:
    """
    """
    #[

    @classmethod
    def from_dataslate(
        klass,
        input_ds: Dataslate,
        num_variants: int,
        name_to_log_name: dict[str, str],
        /,
    ) -> Self:
        """
        """
        self = klass()
        self._dataslate = Dataslate.nan_from_template(input_ds, num_variants=num_variants, )
        self._prepare_log_names(name_to_log_name, )
        return self

    @classmethod
    def from_names_dates(
        klass,
        measurement_names: Iterable[str],
        dates: Iterable[Dater],
        base_columns: Iterable[int],
        num_variants: int,
        name_to_log_name: dict[str, str],
    ) -> Self:
        """
        """
        self = klass()
        self._dataslate = Dataslate.nan_from_names_dates(
            measurement_names, dates,
            base_columns=base_columns,
            num_variants=num_variants,
        )
        self._prepare_log_names(name_to_log_name, )
        return self

    def _prepare_log_names(
        self,
        name_to_log_name: dict[str, str],
        /,
    ) -> None:
        """
        """
        self._dataslate.rename(name_to_log_name, )
        output_names = self._dataslate.output_names
        self._name_to_log_name = {
            name: log_name
            for name, log_name in name_to_log_name.items()
            if log_name in output_names
        }

    def extend(
        self,
        other: Self,
        /,
    ) -> None:
        """
        """
        self._dataslate.extend(other._dataslate, )

    #]


class _MedLogDataslate(_LogDataslate, ):
    """
    """
    #[

    def store(
        self: Self,
        med_array: _np.ndarray,
        lhs_indexes: tuple,
        rhs_indexes=None,
        transform=None,
    ) -> None:
        """
        """
        rhs_indexes = rhs_indexes if rhs_indexes is not None else slice(None, )
        if transform is not None:
            med_array = transform[rhs_indexes, :] @ med_array
        else:
            med_array = med_array[rhs_indexes]
        self._dataslate._variants[0].data[lhs_indexes] = med_array

    def to_output_arg(
        self: Self,
        /,
    ) -> Databox:
        """
        """
        db = self._dataslate.to_databox()
        for name, log_name in self._name_to_log_name.items():
            db[name] = db[log_name].copy()
            db[name].exp()
        return db

    #]


class _StdLogDataslate(_LogDataslate, ):
    """
    """
    #[

    def store(
        self: Self,
        mse_array: _np.ndarray,
        lhs_indexes: tuple,
        rhs_indexes=None,
        transform=None,
    ) -> None:
        rhs_indexes = rhs_indexes if rhs_indexes is not None else slice(None, )
        if transform is not None:
            transform_rhs_indexes = transform[rhs_indexes, :]
            mse_array = transform_rhs_indexes @ mse_array @ transform_rhs_indexes.T
        else:
            mse_array = mse_array[rhs_indexes, ...][..., rhs_indexes]
        self._dataslate._variants[0].data[lhs_indexes] = _std_from_mse(mse_array, )

    def to_output_arg(
        self: Self,
        /,
    ) -> Databox:
        """
        """
        return self._dataslate.to_databox()

    #]


class _OutputStore:
    """
    """
    #[

    __slots__ = (
        "predict_med",
        "predict_std",
        "predict_mse_measurement",

        "update_med",
        "update_std",
        "predict_err",

        "smooth_med",
        "smooth_std",
    )

    def __init__(
        self,
        input_ds: Dataslate,
        num_variants: int,
        measurement_names: Iterable[str],
        needs: _Needs,
        name_to_log_name: dict[str, str] | None = None,
    ) -> Self:
        """
        """
        _med_constructor = _ft.partial(
            _MedLogDataslate.from_dataslate,
            input_ds, num_variants, name_to_log_name,
        )
        _std_constructor = _ft.partial(
            _StdLogDataslate.from_dataslate,
            input_ds, num_variants, name_to_log_name,
        )
        if needs.return_predict:
            self.predict_med = _med_constructor()
            self.predict_std = _std_constructor()
            self.predict_mse_measurement = [ () for _ in range(num_variants, ) ]
        if needs.return_update:
            self.update_med = _med_constructor()
            self.update_std = _std_constructor()
            self.predict_err = _MedLogDataslate.from_names_dates(
                measurement_names,
                input_ds.dates,
                input_ds.base_columns,
                num_variants,
                name_to_log_name,
            )
        if needs.return_smooth:
            self.smooth_med = _med_constructor()
            self.smooth_std = _std_constructor()

    def extend(self, other: Self, /, ) -> None:
        """
        """
        for s in self.__slots__:
            if getattr(self, s) is not None:
                getattr(self, s).extend(getattr(other, s), )

    def rescale_stds(self, var_scale, /, ) -> None:
        """
        """
        if var_scale == 1 or var_scale is None:
            return
        std_scale = _sqrt_positive(self.var_scale)
        for attr_name in ("predict_std", "update_std", "smooth_std", ):
            attr = getattr(self, attr_name, )
            if attr is None:
                continue
            attr._variants = [ v.data * std_scale for v in attr._variants ]

    def create_output_data(self: Self, /, ) -> KalmanOutputData:
        """
        """
        return KalmanOutputData(self, )

    #]


@_dc.dataclass(slots=True, )
class _Cache:
    """
    """
    #[
    LOG_2_PI = _np.log(2 * _np.pi)

    needs: _Needs | None = None

    base_columns: Any | None = None
    nonbase_columns: Any | None = None
    needs: Any | None = None

    all_num_obs: Any | None = None
    all_pex_invFx_pex: Any | None = None
    all_invFx_pex: Any | None = None
    all_det_Fx: Any | None = None

    all_L: Any | None = None
    all_a0: Any | None = None
    all_P0: Any | None = None
    all_ZaxT_invFx_Zax: Any | None = None
    all_ZaxT_invFx: Any | None = None
    all_G: Any | None = None
    all_inx_y: Any | None = None
    all_pex: Any | None = None
    all_invFx: Any | None = None

    all_a1: Any | None = None
    all_P1: Any | None = None

    sum_num_obs: Any | None = None
    sum_log_det_Fx: Any | None = None
    sum_pex_invFx_pex: Any | None = None

    y1_array: Any | None = None
    u0_array: Any | None = None
    v0_array: Any | None = None
    w0_array: Any | None = None

    std_u_array: Any | None = None
    std_w_array: Any | None = None
    all_cov_u: Any | None = None
    all_cov_w: Any | None = None

    var_scale: Any | None = None
    neg_log_likelihood: Any | None = None
    diffuse_factor: Any | None = None

    curr_qids: list[int] | None = None
    curr_pos: list[int] | None = None
    y_qids: list[int] | None = None
    u_qids: list[int] | None = None
    v_qids: list[int] | None = None
    w_qids: list[int] | None = None
    std_u_qids: list[int] | None = None
    std_w_qids: list[int] | None = None

    logly_within_y: list[int] | None = None

    def __init__(
        self: Self,
        input_ds: Dataslate,
        needs: _Needs,
    ) -> None:
        """
        """
        num_ext_periods = input_ds.num_periods
        self.base_columns = input_ds.base_columns
        self.nonbase_columns = input_ds.nonbase_columns
        self.needs = needs
        #
        self.all_num_obs = [None] * num_ext_periods
        self.all_pex_invFx_pex = [None] * num_ext_periods
        self.all_invFx_pex = [None] * num_ext_periods
        self.all_det_Fx = [None] * num_ext_periods
        self.all_L = [None] * num_ext_periods
        self.all_a0 = [None] * num_ext_periods
        self.all_P0 = [None] * num_ext_periods
        self.all_pex = [None] * num_ext_periods
        self.all_invFx = [None] * num_ext_periods
        self.all_ZaxT_invFx = [None] * num_ext_periods
        self.all_ZaxT_invFx_Zax = [None] * num_ext_periods
        self.all_G = [None] * num_ext_periods
        self.all_a1 = [None] * num_ext_periods
        self.all_P1 = [None] * num_ext_periods
        self.all_cov_u = [None] * num_ext_periods
        self.all_cov_w = [None] * num_ext_periods
        #
        self.var_scale = None
        self.neg_log_likelihood = None
        self.diffuse_factor = None
        #
        self.curr_qids = None
        self.curr_pos = None
        self.y_qids = None
        self.u_qids = None
        self.v_qids = None
        self.w_qids = None
        self.std_u_qids = None
        self.std_w_qids = None

    @property
    def reversed_base_columns(self: Self, /, ) -> Any:
        """
        """
        return reversed(self.base_columns, )

    def calculate_likelihood(self: Self, /, ) -> None:
        """
        """
        self.sum_num_obs = sum(i for i in self.all_num_obs if i is not None)
        self.sum_log_det_Fx = sum(i for i in self.all_det_Fx if i is not None)
        self.sum_pex_invFx_pex = sum(i for i in self.all_pex_invFx_pex if i is not None)
        #
        self._calculate_variance_scale()
        #
        self.neg_log_likelihood = (
            + self.sum_num_obs*self.LOG_2_PI
            + self.sum_log_det_Fx
            + self.sum_pex_invFx_pex
        ) / 2;

    def _calculate_variance_scale(self: Self, /, ) -> None:
        """
        """
        if not self.needs.rescale_variance:
            self.var_scale = 1
            return
        if self.sum_num_obs == 0:
            self.var_scale = 1
            self.sum_pex_invFx_pex = 0
            return
        self.var_scale = self.sum_pex_invFx_pex / self.sum_num_obs
        self.sum_log_det_Fx += self.sum_num_obs * _np.log(self.var_scale)
        self.sum_pex_invFx_pex = self.sum_pex_invFx_pex / self.var_scale

    def create_output_info(self: Self, /, ) -> dict[str, Any]:
        """
        """
        return {
            "neg_log_likelihood": self.neg_log_likelihood,
            "var_scale": self.var_scale,
            "diffuse_factor": self.diffuse_factor,
            "cache": self,
        }

    #]


def _sqrt_positive(x):
    return _np.sqrt(_np.maximum(x, 0.0, ))


def _std_from_mse(mse, ):
    return _sqrt_positive(_np.diag(mse).copy())


class Mixin:
    """
    """
    #[

    def kalman_filter(
        self,
        #
        input_db: Databox,
        span: Iterable[Dater],
        #
        diffuse_factor: Real | None = None,
        #
        return_: Iterable[str, ...] = ("predict", "update", "smooth", ),
        return_predict: bool = True,
        return_update: bool = True,
        return_smooth: bool = True,
        rescale_variance: bool = False,
        #
        shocks_from_databox: bool = False,
        stds_from_databox: bool = False,
    ) -> Databox:
        """
        """
        #[

        work_db = input_db.shallow()
        if not shocks_from_databox:
            shock_names = self.get_names(kind=_quantities.ANY_SHOCK, )
            work_db.remove(shock_names, strict_names=False, )
        if not stds_from_databox:
            std_names = self.get_names(kind=_quantities.ANY_STD, )
            work_db.remove(std_names, strict_names=False, )

        input_ds = Dataslate.from_databox_for_slatable(
            self,
            work_db,
            span,
            extend_span=False,
        )

        needs = _Needs(
            return_predict=return_predict and "predict" in return_,
            return_update=return_update and "update" in return_,
            return_smooth=return_smooth and "smooth" in return_,
            rescale_variance=rescale_variance,
        )

        solution_vectors = self._solution_vectors
        temp = [(t.qid, i) for i, t in enumerate(solution_vectors.transition_variables) if not t.shift ]
        curr_qids, curr_pos = tuple(zip(*temp))
        curr_qids = list(curr_qids)
        curr_pos = list(curr_pos)

        y_qids = [ t.qid for t in solution_vectors.measurement_variables ]
        u_qids = [ t.qid for t in solution_vectors.unanticipated_shocks ]
        v_qids = [ t.qid for t in solution_vectors.anticipated_shocks ]
        w_qids = [ t.qid for t in solution_vectors.measurement_shocks ]

        qid_to_logly = self.create_qid_to_logly()
        logly_within_y = tuple( i for i, qid in enumerate(y_qids, ) if qid_to_logly.get(qid, False) )

        std_u_qids = self.get_std_qids_for_shock_qids(u_qids, )
        std_w_qids = self.get_std_qids_for_shock_qids(w_qids, )

        num_variants = 1

        qid_to_name = self.create_qid_to_name()
        y_names = [ qid_to_name[qid] for qid in y_qids ]

        name_to_log_name = {
            n: _quantities.wrap_logly(n, )
            for qid, n in enumerate(input_ds.names, )
            if qid_to_logly.get(qid, False, )
        } if qid_to_logly else {}

        output_store = _OutputStore(
            input_ds=input_ds,
            num_variants=0,
            measurement_names=y_names,
            needs=needs,
            name_to_log_name=name_to_log_name,
        )

        output_info = []


        #
        # Variant dependent from here
        #
        input_ds_v = input_ds.get_variant(0, )
        solution = self.get_solution_matrices()

        output_store_v = _OutputStore(
            input_ds=input_ds,
            num_variants=1,
            measurement_names=y_names,
            needs=needs,
            name_to_log_name=name_to_log_name,
        )

        cache = _Cache(
            input_ds=input_ds,
            needs=needs,
        )

        cache.curr_qids = list(curr_qids)
        cache.curr_pos = list(curr_pos)
        cache.y_qids = list(y_qids)
        cache.u_qids = list(u_qids)
        cache.v_qids = list(v_qids)
        cache.w_qids = list(w_qids)
        cache.std_u_qids = list(std_u_qids)
        cache.std_w_qids = list(std_w_qids)
        cache.logly_within_y = logly_within_y

        #
        # Calculate initial med and mse
        #
        init_cov_u = self.get_cov_unanticipated_shocks()
        init_med, init_mse, cache.diffuse_factor = _initializers.initialize(
            solution,
            init_cov_u,
            diffuse_scale=diffuse_factor,
        )

        # tolerance = 1e-12
        # Za[_np.abs(Za)<tolerance] = 0
        # Ua[_np.abs(Ua)<tolerance] = 0
        # Pa[_np.abs(Pa)<tolerance] = 0

        input_data_array = input_ds_v.get_data_variant()


        a1, P1 = init_med, init_mse
        Ta = solution.Ta
        Ra = solution.Ra
        Pa = solution.Pa
        Ka = solution.Ka
        Za = solution.Za
        H = solution.H
        D = solution.D
        Ua = solution.Ua

        cache.y1_array = input_data_array[cache.y_qids, :]
        cache.u0_array = input_data_array[cache.u_qids, :]
        cache.v0_array = input_data_array[cache.v_qids, :]
        cache.w0_array = input_data_array[cache.w_qids, :]
        cache.std_u_array = input_data_array[cache.std_u_qids, :]
        cache.std_w_array = input_data_array[cache.std_w_qids, :]

        if cache.logly_within_y:
            cache.y1_array[cache.logly_within_y, :] = _np.log(cache.y1_array[cache.logly_within_y, :])

        for n in ("y1_array", "u0_array", "v0_array", "w0_array", ):
            array = getattr(cache, n, )
            array[:, cache.nonbase_columns] = _np.nan
            setattr(cache, n, array, )

        cache.all_inx_y = tuple(
            _np.isfinite(column_data, )
            for column_data in cache.y1_array.T
        )

        if needs.return_predict:
            output_store_v.predict_med.store(cache.u0_array, (cache.u_qids, ...), )
            output_store_v.predict_med.store(cache.v0_array, (cache.v_qids, ...), )
            output_store_v.predict_med.store(cache.w0_array, (cache.w_qids, ...), )

        if needs.return_update:
            output_store_v.update_med.store(cache.y1_array, (cache.y_qids, ...), )

        if needs.return_smooth:
            pass

        for t in cache.base_columns:

            inx_y = cache.all_inx_y[t]

            # Extract time-varying means and stds for shocks
            u0 = cache.u0_array[:, t]
            v0 = cache.v0_array[:, t]
            w0 = cache.w0_array[:, t]
            cov_u = _np.diag(cache.std_u_array[:, t]**2, )
            cov_w = _np.diag(cache.std_w_array[:, t]**2, )

            # MSE prediction step
            P0 = Ta @ P1 @ Ta.T + Pa @ cov_u @ Pa.T
            P0 = (P0 + P0.T) / 2
            F = Za @ P0 @ Za.T + H @ cov_w @ H.T
            F = (F + F.T) / 2
            Fx = F[inx_y, :][:, inx_y]
            if needs.return_predict:
                output_store_v.predict_std.store(P0, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )
                output_store_v.predict_mse_measurement[0] += (Fx, )

            # Mean prediction step
            a0 = Ta @ a1 + Ka + Pa @ u0
            y0 = Za @ a0 + D + H @ w0
            y0[~inx_y] = _np.nan

            if needs.return_predict:
                output_store_v.predict_med.store(a0, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )
                output_store_v.predict_med.store(y0, (cache.y_qids, t), )
                output_store_v.predict_med.store(u0, (cache.u_qids, t), )
                output_store_v.predict_med.store(v0, (cache.v_qids, t), )
                output_store_v.predict_med.store(w0, (cache.w_qids, t), )

            # MSE updating step
            Zax = Za[inx_y, :]
            # ZaxT_invFx = right_div(Zax.T, Fx, )
            # if Fx.size > 0:
            #     sing_values = _np.linalg.svd(Fx, compute_uv=False, )
            #     relative_cond = sing_values[-1] / sing_values[0]
            #     print(t, Fx.size, relative_cond, sing_values[0], )
            invFx = _np.linalg.inv(Fx)
            ZaxT_invFx = Zax.T @ invFx
            G = P0 @ ZaxT_invFx
            P1 = P0 - G @ Zax @ P0
            P1 = (P1 + P1.T) / 2

            if needs.return_update:
                output_store_v.update_std.store(P1, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )

            # Mean updating step
            cache.all_num_obs[t] = _np.count_nonzero(inx_y, )
            y1 = cache.y1_array[:, t]
            pe = y1 - y0
            pex = pe[inx_y]
            a1 = a0 + G @ pex

            # invFx_pex = left_div(Fx, pex, ) # inv(Fx) * pex
            invFx_pex = invFx @ pex # inv(Fx) * pex

            if needs.return_update:
                Hx_cov_w = H[inx_y, :] @ cov_w
                w1 = w0 + Hx_cov_w.T @ invFx_pex
                Pa_cov_u = Pa @ cov_u
                r = ZaxT_invFx @ pex # Z' * inv(F) * pe
                u1 = u0 + Pa_cov_u.T @ r
                v1 = v0
                output_store_v.update_med.store(a1, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )
                output_store_v.update_med.store(y1, (cache.y_qids, t), )
                output_store_v.update_med.store(u1, (cache.u_qids, t), )
                output_store_v.update_med.store(v1, (cache.v_qids, t), )
                output_store_v.update_med.store(w1, (cache.w_qids, t), )
                output_store_v.predict_err.store(pe, (..., t), )

            if needs.return_smooth:
                cache.all_a0[t] = a0
                cache.all_P0[t] = P0
                cache.all_a1[t] = a1
                cache.all_P1[t] = P1
                cache.all_invFx_pex[t] = invFx_pex
                cache.all_ZaxT_invFx[t] = Zax.T @ invFx
                cache.all_ZaxT_invFx_Zax[t] = ZaxT_invFx @ Zax
                cache.all_L[t] = Ta - (Ta @ G) @ Zax
                cache.all_G[t] = G
                cache.all_cov_u[t] = cov_u
                cache.all_cov_w[t] = cov_w

            cache.all_pex[t] = pex
            cache.all_invFx[t] = invFx
            cache.all_pex_invFx_pex[t] = pex @ invFx_pex
            cache.all_det_Fx[t] = _np.log(_np.linalg.det(Fx))


        if needs.return_smooth:
            _smooth_backward(cache, output_store_v, needs, solution, )

        cache.calculate_likelihood()
        output_store_v.rescale_stds(cache.var_scale, )

        output_store.extend(output_store_v, )
        output_info.append(cache.create_output_info(), )

        return output_store.create_output_data(), output_info

    #]


def _smooth_backward(
    cache: _Cache,
    output_store_v: _OutputStore,
    needs: _Needs,
    solution: _simultaneous.Solution,
) -> None:
    """
    """
    #[

    Ta = solution.Ta
    Ua = solution.Ua
    Pa = solution.Pa
    H = solution.H

    N = None
    r = None

    for t in cache.reversed_base_columns:

        inx_y = cache.all_inx_y[t]
        u0 = cache.u0_array[:, t]
        v0 = cache.v0_array[:, t]
        w0 = cache.w0_array[:, t]
        cov_u = cache.all_cov_u[t]
        cov_w = cache.all_cov_w[t]

        a0 = cache.all_a0[t]
        L = cache.all_L[t]
        G = cache.all_G[t]
        P0 = cache.all_P0[t]
        ZaxT_invFx = cache.all_ZaxT_invFx[t]
        ZaxT_invFx_Zax = cache.all_ZaxT_invFx_Zax[t]
        invFx = cache.all_invFx[t]
        pex = cache.all_pex[t]

        N = ZaxT_invFx_Zax + ((L.T @ N @ L) if N is not None else 0)
        P2 = P0 - P0 @ N @ P0
        output_store_v.smooth_std.store(P2, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )

        H_cov_w = H[inx_y, :] @ cov_w
        w2 = w0 + H_cov_w.T @ (invFx @ pex - (((Ta @ G).T @ r) if r is not None else 0))
        output_store_v.smooth_med.store(w2, (cache.w_qids, t), )

        r = ZaxT_invFx @ pex + ((L.T @ r) if r is not None else 0)
        a2 = a0 + P0 @ r
        output_store_v.smooth_med.store(a2, (cache.curr_qids, t), rhs_indexes=cache.curr_pos, transform=Ua, )

        y2 = cache.y1_array[:, t]
        output_store_v.smooth_med.store(y2, (cache.y_qids, t), )

        Pa_cov_u = Pa @ cov_u
        u2 = u0 + Pa_cov_u.T @ r
        output_store_v.smooth_med.store(u2, (cache.u_qids, t), )

        v2 = v0
        output_store_v.smooth_med.store(v2, (cache.v_qids, t), )

    #]

