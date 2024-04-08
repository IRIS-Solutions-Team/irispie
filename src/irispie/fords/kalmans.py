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

from .. import has_variants as _has_variants
from .. import quantities as _quantities
from .. import pages as _pages
from ..dataslates.main import (Dataslate, )
from ..databoxes.main import (Databox, )
from .solutions import (right_div, left_div, )
from ..dates import (Dater, )

from . import initializers as _ford_initializers
from . import simulators as _ford_simulators
#]


simulate_anticipated_shocks \
    = _ford_simulators.simulate_triangular_anticipated_shocks


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
        num_variants: int,
        name_to_log_name: dict[str, str],
    ) -> Self:
        """
        """
        self = klass()
        self._dataslate = Dataslate.nan_from_names_dates(
            measurement_names, dates,
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


class _Cache:
    """
    """
    #[

    LOG_2_PI = _np.log(2 * _np.pi)

    __slots_to_input__ = (
        "model_v",
        "input_ds_v",
        "solution",
        "needs",
        "curr_qids",
        "curr_pos",
        "y_qids",
        "u_qids",
        "v_qids",
        "w_qids",
        "std_u_qids",
        "std_w_qids",
        "logly_within_y",
    )

    __slots_to_preallocate__ = (
        "all_num_obs",
        "all_pex_invFx_pex",
        "all_invFx_pex",
        "all_log_det_Fx",
        "all_L",
        "all_a0",
        "all_P0",
        "all_ZaxT_invFx_Zax",
        "all_ZaxT_invFx",
        "all_G",
        "all_inx_y",
        "all_pex",
        "all_invFx",
        "all_a1",
        "all_P1",
        "all_cov_u",
        "all_cov_w",
    )

    __slots_to_create__ = (
        "num_periods",
        "base_columns",
        "nonbase_columns",
        "sum_num_obs",
        "sum_log_det_Fx",
        "sum_pex_invFx_pex",
        "y1_array",
        "u0_array",
        "v0_array",
        "w0_array",
        "all_v_impact",
        "std_u_array",
        "std_w_array",
        "var_scale",
        "neg_log_likelihood",
        "neg_log_likelihood_contributions",
        "diffuse_factor",
    )

    __slots__ = __slots_to_input__ + __slots_to_preallocate__ + __slots_to_create__

    def __init__(self: Self, **kwargs, ) -> None:
        """
        """
        for n in self.__slots_to_input__:
            setattr(self, n, kwargs[n], )
        #
        self.num_periods = self.input_ds_v.num_periods
        self.base_columns = self.input_ds_v.base_columns
        self.nonbase_columns = self.input_ds_v.nonbase_columns
        #
        # Preallocate time-varying arrayas
        for n in self.__slots_to_preallocate__:
            setattr(self, n, [None] * self.num_periods, )
        #
        # Extract arrays from input dataslate
        input_data_array = self.input_ds_v.get_data_variant(0, )
        self.y1_array = input_data_array[self.y_qids, :]
        self.u0_array = input_data_array[self.u_qids, :]
        self.v0_array = input_data_array[self.v_qids, :]
        self.w0_array = input_data_array[self.w_qids, :]
        self.std_u_array = input_data_array[self.std_u_qids, :]
        self.std_w_array = input_data_array[self.std_w_qids, :]
        if self.logly_within_y:
            self.y1_array[self.logly_within_y, :] = _np.log(self.y1_array[self.logly_within_y, :])
        #
        # Detect observations available in each period
        self.all_inx_y = [
            _np.isfinite(column_data, )
            for column_data in self.y1_array.T
        ]
        # Simulate impact of anticipated shocks
        self.all_v_impact = simulate_anticipated_shocks(
            self.model_v,
            self.input_ds_v,
        )

    def calculate_likelihood(self: Self, /, ) -> None:
        """
        """
        self.sum_num_obs = sum(i for i in self.all_num_obs if i is not None)
        self.sum_log_det_Fx = sum(i for i in self.all_log_det_Fx if i is not None)
        self.sum_pex_invFx_pex = sum(i for i in self.all_pex_invFx_pex if i is not None)
        #
        self._calculate_variance_scale()
        #
        self.neg_log_likelihood_contributions = [
            (log_det_Fx + pex_invFx_pex + num_obs * self.LOG_2_PI)/2 if num_obs else 0
            for log_det_Fx, pex_invFx_pex, num_obs in zip(
                self.all_log_det_Fx, self.all_pex_invFx_pex, self.all_num_obs,
            )
            if num_obs is not None
        ]
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
            "neg_log_likelihood_contributions": self.neg_log_likelihood_contributions,
            "var_scale": self.var_scale,
            "diffuse_factor": self.diffuse_factor,
            "anticipated_shocks_impact": self.all_v_impact,
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

    @_pages.reference(category="filtering", )
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
        shocks_from_data: bool = False,
        stds_from_data: bool = False,
        #
        prepend_initial: bool = False,
        append_terminal: bool = False,
        #
        unpack_singleton: bool = True,
    ) -> Databox:
        r"""
················································································

==Run Kalman filter on a model using time series data==

Executes a Kalman filter on a model, compliant with `KalmanFilterableProtocol`, 
using time series observations from the input Databox. This method enables state 
estimation and uncertainty quantification in line with the model's dynamics and 
the time series data.

    kalman_output, output_info = self.kalman_filter(
        input_db, 
        span, 
        diffuse_factor=None, 
        return_=("predict", "update", "smooth"),
        return_predict=True, 
        return_update=True, 
        return_smooth=True, 
        rescale_variance=False,
        shocks_from_data=False, 
        stds_from_data=False, 
        prepend_initial=False,
        append_terminal=False, 
        unpack_singleton=True
    )


### Input arguments ###


???+ input "self"
    The model, compliant with `KalmanFilterableProtocol`, performing the 
    Kalman filtering.

???+ input "input_db"
    A Databox containing time series data to be used for filtering.

???+ input "span"
    A date span over which the filtering process is executed based on the
    measurement time series.

???+ input "diffuse_factor"
    A real number or `None`, specifying the scale factor for the diffuse
    initialization. If `None`, the default value is used.

???+ input "return_"
    An iterable of strings indicating which steps' results to return: 
    "predict", "update", "smooth".

???+ input "return_predict"
    If `True`, return prediction step results.

???+ input "return_update"
    If `True`, return update step results.

???+ input "return_smooth"
    If `True`, return smoothing step results.

???+ input "rescale_variance"
    If `True`, rescale all variances by the optimal variance scale factor
    estimated using maximum likelihood after the filtering process.

???+ input "shocks_from_data"
    If `True`, use possibly time-varying shock values from the data; these
    values are interpreted as the medians (means) of the shocks. If `False`,
    zeros are used for all shocks.

???+ input "stds_from_data"
    If `True`, use possibly time-varying standard deviation values from the
    data. If `False`, currently assigned constant values are used for the
    standard deviations of all shocks.

???+ input "prepend_initial"
    If `True`, prepend observations to the resulting time series to cover
    initial conditions based on the model's maximum lag. No measurement
    observations are used in these initial time periods (even if some are
    available in the input data).

???+ input "append_terminal"
    If `True`, append observations to the resulting time series to cover
    terminal conditions based on the model's maximum lead. No measurement
    observations are used in these terminal time periods (even if some are
    available in the input data).

???+ input "unpack_singleton"
    If `True`, unpack `output_info` into a plain dictionary for models with a
    single variant.


### Returns ###


???+ returns "kalman_output"
    An object containing the following attributes, each being a Databox:

    | Attribute                  | Description
    |----------------------------|---------------------------------------------------
    | `predict_med`              | Medians from the prediction step.
    | `predict_std`              | Standard deviations from the prediction step.
    | `predict_mse_measurement`  | Mean squared error matrices from the prediction step.
    | `update_med`               | Medians from the update step.
    | `update_std`               | Standard deviations from the update step.
    | `predict_err`              | Prediction errors.
    | `smooth_med`               | Medians from the smoothing step.
    | `smooth_std`               | Standard deviations from the smoothing step.

    Some of these attributes may be `None` if the corresponding step was not
    requested in `return_`.

???+ returns "output_info"
    A dictionary containing additional information about the filtering process,
    such as log likelihood and variance scale. For models with multiple
    variants, `output_info` is a list of such dictionaries. If
    `unpack_singleton=False`, also `output_info` is a one-element list
    containing the dictionary for singleton models, too.

················································································
        """
        work_db = input_db.shallow()
        if not shocks_from_data:
            shock_names = self.get_names(kind=_quantities.ANY_SHOCK, )
            work_db.remove(shock_names, strict_names=False, )
        if not stds_from_data:
            std_names = self.get_names(kind=_quantities.ANY_STD, )
            work_db.remove(std_names, strict_names=False, )

        slatable = self.get_slatable(
            shocks_from_data=shocks_from_data,
            stds_from_data=stds_from_data,
        )

        input_ds = Dataslate.from_databox_for_slatable(
            slatable, work_db, span,
            prepend_initial=prepend_initial,
            append_terminal=append_terminal,
            clip_data_to_base_span=True,
        )

        needs = _Needs(
            return_predict=return_predict and "predict" in return_,
            return_update=return_update and "update" in return_,
            return_smooth=return_smooth and "smooth" in return_,
            rescale_variance=rescale_variance,
        )

        solution_vectors = self.solution_vectors
        qid_pos_tuples = [
            (tok.qid, pos)
            for pos, tok in enumerate(solution_vectors.transition_variables, )
            if not tok.shift
        ]
        curr_qids, curr_pos = tuple(zip(*qid_pos_tuples))
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

        create_cache = _ft.partial(
            _Cache,
            needs=needs,
            curr_qids=curr_qids,
            curr_pos=curr_pos,
            y_qids=y_qids,
            u_qids=u_qids,
            v_qids=v_qids,
            w_qids=w_qids,
            std_u_qids=std_u_qids,
            std_w_qids=std_w_qids,
            logly_within_y=logly_within_y,
        )


        # TODO: Iterate over variants

        #
        # Variant dependent from here
        #
        input_ds_v = input_ds.get_variant(0, )
        self_v = self.get_variant(0, )

        output_store_v = _OutputStore(
            input_ds=input_ds_v,
            num_variants=1,
            measurement_names=y_names,
            needs=needs,
            name_to_log_name=name_to_log_name,
        )

        cache_v = create_cache(
            model_v=self_v,
            input_ds_v=input_ds_v,
            solution=self_v.get_solution(),
        )

        #
        # Initialize medians and stds
        #
        init_cov_u = self_v.get_cov_unanticipated_shocks()
        init_med, init_mse, cache_v.diffuse_factor = _ford_initializers.initialize(
            cache_v.solution,
            init_cov_u,
            diffuse_scale=diffuse_factor,
        )

        # tolerance = 1e-12
        # Za[_np.abs(Za)<tolerance] = 0
        # Ua[_np.abs(Ua)<tolerance] = 0
        # Pa[_np.abs(Pa)<tolerance] = 0

        a1, P1 = init_med, init_mse
        Ta = cache_v.solution.Ta
        Ra = cache_v.solution.Ra
        Pa = cache_v.solution.Pa
        Ka = cache_v.solution.Ka
        Za = cache_v.solution.Za
        H = cache_v.solution.H
        D = cache_v.solution.D
        Ua = cache_v.solution.Ua


        if needs.return_predict:
            output_store_v.predict_med.store(cache_v.u0_array, (cache_v.u_qids, ...), )
            output_store_v.predict_med.store(cache_v.v0_array, (cache_v.v_qids, ...), )
            output_store_v.predict_med.store(cache_v.w0_array, (cache_v.w_qids, ...), )

        if needs.return_update:
            output_store_v.update_med.store(cache_v.y1_array, (cache_v.y_qids, ...), )

        if needs.return_smooth:
            pass

        for t in range(cache_v.num_periods, ):

            inx_y = cache_v.all_inx_y[t]

            # Extract time-varying means and stds for shocks
            u0 = cache_v.u0_array[:, t]
            v0 = cache_v.v0_array[:, t]
            w0 = cache_v.w0_array[:, t]
            cov_u = _np.diag(cache_v.std_u_array[:, t]**2, )
            cov_w = _np.diag(cache_v.std_w_array[:, t]**2, )
            v_impact = cache_v.all_v_impact[t] if cache_v.all_v_impact[t] is not None else 0

            #
            # MSE prediction step
            #
            P0 = Ta @ P1 @ Ta.T + Pa @ cov_u @ Pa.T
            P0 = (P0 + P0.T) / 2
            F = Za @ P0 @ Za.T + H @ cov_w @ H.T
            F = (F + F.T) / 2
            Fx = F[inx_y, :][:, inx_y]
            if needs.return_predict:
                output_store_v.predict_std.store(P0, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )
                output_store_v.predict_mse_measurement[0] += (Fx, )

            #
            # Mean prediction step
            #
            a0 = Ta @ a1 + Ka + Pa @ u0 + v_impact
            y0 = Za @ a0 + D + H @ w0
            y0[~inx_y] = _np.nan

            if needs.return_predict:
                output_store_v.predict_med.store(a0, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )
                output_store_v.predict_med.store(y0, (cache_v.y_qids, t), )
                output_store_v.predict_med.store(u0, (cache_v.u_qids, t), )
                output_store_v.predict_med.store(v0, (cache_v.v_qids, t), )
                output_store_v.predict_med.store(w0, (cache_v.w_qids, t), )

            #
            # MSE updating step
            #
            if Fx.size:
                # sing_values = _np.linalg.svd(Fx, compute_uv=False, )
                # relative_cond = sing_values[-1] / sing_values[0]
                # print(t, Fx.size, relative_cond, sing_values[0], )
                # TODO: Check if Fx singularity if requested by user
                invFx = _np.linalg.inv(Fx)
            else:
                invFx = _np.zeros((0, 0), dtype=_np.float64, )
            #
            Zax = Za[inx_y, :]
            # ZaxT_invFx = right_div(Zax.T, Fx, )
            #
            ZaxT_invFx = Zax.T @ invFx
            G = P0 @ ZaxT_invFx
            P1 = P0 - G @ Zax @ P0
            P1 = (P1 + P1.T) / 2

            if needs.return_update:
                output_store_v.update_std.store(P1, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )

            #
            # Mean updating step
            #
            cache_v.all_num_obs[t] = _np.count_nonzero(inx_y, )
            y1 = cache_v.y1_array[:, t]
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
                output_store_v.update_med.store(a1, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )
                output_store_v.update_med.store(y1, (cache_v.y_qids, t), )
                output_store_v.update_med.store(u1, (cache_v.u_qids, t), )
                output_store_v.update_med.store(v1, (cache_v.v_qids, t), )
                output_store_v.update_med.store(w1, (cache_v.w_qids, t), )
                output_store_v.predict_err.store(pe, (..., t), )

            if needs.return_smooth:
                cache_v.all_a0[t] = a0
                cache_v.all_P0[t] = P0
                cache_v.all_a1[t] = a1
                cache_v.all_P1[t] = P1
                cache_v.all_invFx_pex[t] = invFx_pex
                cache_v.all_ZaxT_invFx[t] = Zax.T @ invFx
                cache_v.all_ZaxT_invFx_Zax[t] = ZaxT_invFx @ Zax
                cache_v.all_L[t] = Ta - (Ta @ G) @ Zax
                cache_v.all_G[t] = G
                cache_v.all_cov_u[t] = cov_u
                cache_v.all_cov_w[t] = cov_w

            cache_v.all_pex[t] = pex
            cache_v.all_invFx[t] = invFx
            cache_v.all_pex_invFx_pex[t] = pex @ invFx_pex
            cache_v.all_log_det_Fx[t] = _np.log(_np.linalg.det(Fx))


        if needs.return_smooth:
            _smooth_backward(cache_v, output_store_v, )

        cache_v.calculate_likelihood()
        output_store_v.rescale_stds(cache_v.var_scale, )
        output_info_v = cache_v.create_output_info()

        output_store.extend(output_store_v, )
        output_info.append(output_info_v, )

        output_info = _has_variants.unpack_singleton(
            output_info, self.is_singleton,
            unpack_singleton=True,
        )

        return output_store.create_output_data(), output_info

    #]


def _smooth_backward(
    cache_v: _Cache,
    output_store_v: _OutputStore,
) -> None:
    """
    """
    #[

    Ta = cache_v.solution.Ta
    Ua = cache_v.solution.Ua
    Pa = cache_v.solution.Pa
    H = cache_v.solution.H

    N = None
    r = None

    for t in reversed(range(cache_v.num_periods, )):

        inx_y = cache_v.all_inx_y[t]
        u0 = cache_v.u0_array[:, t]
        v0 = cache_v.v0_array[:, t]
        w0 = cache_v.w0_array[:, t]
        cov_u = cache_v.all_cov_u[t]
        cov_w = cache_v.all_cov_w[t]

        a0 = cache_v.all_a0[t]
        L = cache_v.all_L[t]
        G = cache_v.all_G[t]
        P0 = cache_v.all_P0[t]
        ZaxT_invFx = cache_v.all_ZaxT_invFx[t]
        ZaxT_invFx_Zax = cache_v.all_ZaxT_invFx_Zax[t]
        invFx = cache_v.all_invFx[t]
        pex = cache_v.all_pex[t]

        N = ZaxT_invFx_Zax + ((L.T @ N @ L) if N is not None else 0)
        P2 = P0 - P0 @ N @ P0
        output_store_v.smooth_std.store(P2, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )

        H_cov_w = H[inx_y, :] @ cov_w
        w2 = w0 + H_cov_w.T @ (invFx @ pex - (((Ta @ G).T @ r) if r is not None else 0))
        output_store_v.smooth_med.store(w2, (cache_v.w_qids, t), )

        r = ZaxT_invFx @ pex + ((L.T @ r) if r is not None else 0)
        a2 = a0 + P0 @ r
        output_store_v.smooth_med.store(a2, (cache_v.curr_qids, t), rhs_indexes=cache_v.curr_pos, transform=Ua, )

        y2 = cache_v.y1_array[:, t]
        output_store_v.smooth_med.store(y2, (cache_v.y_qids, t), )

        Pa_cov_u = Pa @ cov_u
        u2 = u0 + Pa_cov_u.T @ r
        output_store_v.smooth_med.store(u2, (cache_v.u_qids, t), )

        v2 = v0
        output_store_v.smooth_med.store(v2, (cache_v.v_qids, t), )

    #]

