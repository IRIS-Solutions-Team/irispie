"""
"""


#[

from __future__ import annotations

from typing import Protocol
from typing import Literal
import functools as _ft
import numpy as _np

from .. import has_variants as _has_variants
from .. import quantities as _quantities
from ..frames import Frame
from ..dataslates.main import Dataslate
from ..databoxes.main import Databox
from ..series.main import Series
from .solutions import Solution, right_div, left_div

from . import initializers as _initializers
from . import shock_simulators as _shock_simulators
from . import covariances as _covariances
from .descriptors import Squid, SquidableProtocol
from .. import wrongdoings as _wd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Iterable, Sequence, Self, NoReturn
    from types import EllipsisType
    from numbers import Real
    from ..dates import Period
    from ..dataslates.slatables import Slatable

#]


_simulate_anticipated_shocks = (
    _shock_simulators
    .simulate_triangular_anticipated_shocks
)


_DEFAULT_DELTA_TOLERANCE = 1e-12


DiffuseMethodType = Literal[
    "approx_diffuse",
    "fixed_unknown",
    "fixed_zero",
]


class SingularMatrixError(ValueError, ):
    pass


class KalmanFilterableProtocol(SquidableProtocol, ):
    """
    """
    #[

    is_singleton: bool

    def resolve_num_variants_in_context(self, num_variants: int, ) -> int: ...

    def slatable_for_kalman_filter(self, shock_from_data: bool = False, stds_from_data: bool = False, ) -> Slatable: ...

    def create_qid_to_logly(self, ) -> dict[int, bool] | None: ...

    def create_qid_to_name(self, ) -> dict[int, str] | None: ...

    def iter_variants(self, ) -> Iterable[Self]: ...

    def _gets_solution(self, deviation: bool = False, ) -> Solution: ...

    def _gets_cov_unanticipated_shocks(self, ) -> _np.ndarray: ...

    #]


class KalmanOutputData:
    """
......................................................................

==Output data from Kalman filter==


......................................................................
    """
    #[

    __slots__ = (
        "predict_mse_obs",
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
                self_attr = attr.to_output_arg()
            else:
                self_attr = attr
            setattr(self, attr_name, self_attr, )

    def __repr__(self, /, ) -> str:
        """
        """
        slots_to_repr = (
            i for i in self.__slots__
            if getattr(self, s) is not None
        )
        return f"KalmanOutputData({', '.join(slots_to_repr)})"

    #]


class Needs:
    """
    """
    #[

    __slots__ = (
        "return_predict",
        "return_update",
        "return_smooth",
        "return_predict_err",
        "return_predict_mse_obs",
        "rescale_variance",
        "likelihood_contributions",
        "output_store",
    )

    def __init__(
        self: Self,
        return_: Iterable[str, ...] = (),
        return_predict: bool = False,
        return_update: bool = False,
        return_smooth: bool = False,
        return_predict_err: bool = False,
        return_predict_mse_obs: bool = False,
        rescale_variance: bool = False,
        likelihood_contributions: bool = False,
    ) -> None:
        """
        """
        return_ = tuple(return_) if not isinstance(return_, str) else (return_, )
        self.return_predict = return_predict and "predict" in return_
        self.return_update = return_update and "update" in return_
        self.return_smooth = return_smooth and "smooth" in return_
        self.return_predict_err = return_predict_err and "predict_err" in return_
        self.return_predict_mse_obs = return_predict_mse_obs and "predict_mse_obs" in return_
        self.rescale_variance = rescale_variance
        self.likelihood_contributions = likelihood_contributions
        #
        self.output_store = any((
            self.return_predict,
            self.return_update,
            self.return_smooth,
        ))

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
    ) -> Self | None:
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
        dates: Iterable[Period],
        num_variants: int,
        name_to_log_name: dict[str, str],
    ) -> Self:
        """
        """
        self = klass()
        self._dataslate = Dataslate.nan_from_names_periods(
            measurement_names, dates,
            num_variants=num_variants,
        )
        self._prepare_log_names(name_to_log_name, )
        return self

    def store(
        self: Self,
        array: _np.ndarray,
        lhs_indexes: tuple,
        rhs_indexes=None,
        transform=None,
    ) -> None:
        """
        """
        if array is None:
            return
        rhs_indexes = rhs_indexes if rhs_indexes is not None else slice(None, )
        if transform is not None:
            array = transform[rhs_indexes, :] @ array
        else:
            array = array[rhs_indexes]
        self._dataslate._variants[0].data[lhs_indexes] = array

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

    def store_from_mse(
        self: Self,
        mse_array: _np.ndarray,
        lhs_indexes: tuple,
        rhs_indexes=None,
        transform=None,
    ) -> None:
        if mse_array is None:
            return
        rhs_indexes = rhs_indexes if rhs_indexes is not None else slice(None, )
        if transform is not None:
            transform_rhs_indexes = transform[rhs_indexes, :]
            mse_array = transform_rhs_indexes @ mse_array @ transform_rhs_indexes.T
        else:
            mse_array = mse_array[rhs_indexes, ...][..., rhs_indexes]
        self._dataslate._variants[0].data[lhs_indexes] = _covariances.std_from_cov(mse_array, )

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

    __slots_to_input__ = (
        "transform",
        "squid",
        "measurement_incidence",
    )

    __slots_to_create__ = (
        "predict_med",
        "predict_std",
        "predict_mse_obs",

        "update_med",
        "update_std",
        "predict_err",

        "smooth_med",
        "smooth_std",
    )

    __slots__ = __slots_to_input__ + __slots_to_create__

    _OUT_NAMES = (
        "predict_mse_obs",
        "predict_med",
        "predict_std",
        "predict_err",
        "update_med",
        "update_std",
        "smooth_med",
        "smooth_std",
    )

    def __init__(
        self,
        input_ds: Dataslate,
        num_variants: int,
        measurement_names: Iterable[str],
        needs: Needs,
        name_to_log_name: dict[str, str] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        #
        for n in self.__slots_to_input__:
            setattr(self, n, kwargs.get(n, ), )
        #
        _med_constructor = _ft.partial(
            _MedLogDataslate.from_dataslate,
            input_ds=input_ds,
            num_variants=num_variants,
            name_to_log_name=name_to_log_name,
        )
        #
        _std_constructor = _ft.partial(
            _StdLogDataslate.from_dataslate,
            input_ds=input_ds,
            num_variants=num_variants,
            name_to_log_name=name_to_log_name,
        )
        #
        # Prediction step
        #
        self.predict_med = _med_constructor() if needs.return_predict else None
        self.predict_std = _std_constructor() if needs.return_predict else None
        self.predict_mse_obs = [
            [None]*input_ds.num_periods for _ in range(num_variants, )
        ] if needs.return_predict and needs.return_predict_mse_obs else None
        #
        # Updating step
        #
        self.update_med = _med_constructor() if needs.return_update else None
        self.update_std = _std_constructor() if needs.return_update else None
        self.predict_err = _MedLogDataslate.from_names_dates(
            measurement_names,
            input_ds.periods,
            num_variants,
            name_to_log_name,
        ) if needs.return_predict_err else None
        #
        # Smoothing step
        #
        self.smooth_med = _med_constructor() if needs.return_smooth else None
        self.smooth_std = _std_constructor() if needs.return_smooth else None

    def extend(self, other: Self, /, ) -> None:
        """
        """
        for s in self.__slots__:
            if getattr(self, s) is not None:
                getattr(self, s).extend(getattr(other, s), )

    def rescale_stds(self, var_scale, /, ) -> None:
        r"""
        """
        if var_scale == 1 or var_scale is None:
            return
        std_scale = _covariances.sqrt_positive(var_scale)
        for attr_name in ("predict_std", "update_std", "smooth_std", ):
            attr = getattr(self, attr_name, )
            if attr is None:
                continue
            for v in attr._dataslate._variants:
                v.rescale_data(std_scale)

    def create_out_data(self: Self, /, ) -> Databox:
        r"""
        """
        out_data = Databox()
        for name in self._OUT_NAMES:
            attr = getattr(self, name, )
            if attr is None:
                continue
            if hasattr(attr, "to_output_arg", ):
                out_data[name] = attr.to_output_arg()
            else:
                out_data[name] = attr
        return out_data

    def store_predict(
        self: Self,
        t: int | EllipsisType,
        inx_y: _np.ndarray | None = None,
        a0: _np.ndarray | None = None,
        u0: _np.ndarray | None = None,
        v0: _np.ndarray | None = None,
        y0: _np.ndarray | None = None,
        w0: _np.ndarray | None = None,
        Q0: _np.ndarray | None = None,
        F: _np.ndarray | None = None,
        cov_u0: _np.ndarray | None = None,
        cov_w0: _np.ndarray | None = None,
    ) -> None:
        r"""
        """
        if self.predict_med is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            u_qids = self.squid.u_qids
            v_qids = self.squid.v_qids
            w_qids = self.squid.w_qids
            y_qids = self.squid.y_qids
            #
            self.predict_med.store(a0, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )
            self.predict_med.store(u0, (u_qids, t), )
            self.predict_med.store(v0, (v_qids, t), )
            self.predict_med.store(w0, (w_qids, t), )
            full_y0 = self._expand_y_to_full(y0, t, )
            self.predict_med.store(full_y0, (y_qids, t), )
            #
        if self.predict_std is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            u_qids = self.squid.u_qids
            w_qids = self.squid.w_qids
            self.predict_std.store_from_mse(Q0, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )
            self.predict_std.store_from_mse(cov_u0, (u_qids, t), )
            self.predict_std.store_from_mse(cov_w0, (w_qids, t), )
            #
        if self.predict_mse_obs is not None and F is not None:
            self.predict_mse_obs[0][t] = F

    def store_update(
        self: Self,
        t: int | EllipsisType,
        inx_y: _np.ndarray | None = None,
        xi: _np.ndarray | None = None,
        u: _np.ndarray | None = None,
        v: _np.ndarray | None = None,
        w: _np.ndarray | None = None,
        y: _np.ndarray | None = None,
        pe: _np.ndarray | None = None,
        Q: _np.ndarray | None = None,
    ) -> None:
        """
        """
        if self.update_med is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            u_qids = self.squid.u_qids
            v_qids = self.squid.v_qids
            w_qids = self.squid.w_qids
            y_qids = self.squid.y_qids
            self.update_med.store(xi, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )
            self.update_med.store(u, (u_qids, t), )
            self.update_med.store(v, (v_qids, t), )
            self.update_med.store(w, (w_qids, t), )
            full_y = self._expand_y_to_full(y, t, )
            self.update_med.store(full_y, (y_qids, t), )
        if self.update_std is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            self.update_std.store_from_mse(Q, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )
        if self.predict_err is not None:
            full_pe = self._expand_y_to_full(pe, t, )
            self.predict_err.store(full_pe, (..., t), )

    def store_smooth(
        self: Self,
        t: int | EllipsisType,
        inx_y: _np.ndarray | None = None,
        xi: _np.ndarray | None = None,
        u: _np.ndarray | None = None,
        v: _np.ndarray | None = None,
        w: _np.ndarray | None = None,
        y: _np.ndarray | None = None,
        Q: _np.ndarray | None = None,
    ) -> None:
        """
        """
        if self.smooth_med is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            u_qids = self.squid.u_qids
            v_qids = self.squid.v_qids
            w_qids = self.squid.w_qids
            y_qids = self.squid.y_qids
            self.smooth_med.store(xi, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )
            self.smooth_med.store(u, (u_qids, t), )
            self.smooth_med.store(v, (v_qids, t), )
            self.smooth_med.store(w, (w_qids, t), )
            full_y = self._expand_y_to_full(y, t, )
            self.smooth_med.store(full_y, (y_qids, t), )
        if self.smooth_std is not None:
            curr_xi_qids = self.squid.curr_xi_qids
            curr_xi_indexes = self.squid.curr_xi_indexes
            self.smooth_std.store_from_mse(Q, (curr_xi_qids, t), rhs_indexes=curr_xi_indexes, transform=self.transform, )

    def _expand_y_to_full(
        self,
        observed: _np.ndarray | None,
        t: int | EllipsisType,
    ) -> _np.ndarray:
        """
        """
        if observed is None or t is Ellipsis:
            return observed
        inx_y = self.measurement_incidence[:, t]
        full = _np.full(inx_y.shape, _np.nan, )
        full[inx_y] = observed
        return full

    #]


def neg_log_likelihood(model, *args, **kwargs, ) -> Real:
    r"""
    """
    #[
    kwargs["return_"] = ()
    kwargs["likelihood_contributions"] = False
    kwargs["return_info"] = True
    _, info = kalman_filter(model, *args, **kwargs, )
    return info["neg_log_likelihood"]
    #]


def kalman_filter(
    model,
    #
    input_db: Databox,
    span: Iterable[Period],
    #
    generate_period_system: Callable,
    generate_period_data: Callable,
    #
    diffuse_scale: Real | None = None,
    diffuse_method: DiffuseMethodType = "fixed_unknown",
    #
    return_: Iterable[str, ...] = ("predict", "update", "smooth", "predict_err", "predict_mse_obs", ),
    return_predict: bool = True,
    return_update: bool = True,
    return_smooth: bool = True,
    return_predict_err: bool = True,
    return_predict_mse_obs: bool = True,
    rescale_variance: bool = False,
    likelihood_contributions: bool = True,
    #
    shocks_from_data: bool = False,
    stds_from_data: bool = False,
    initials_from_data: bool = False,
    output_parameters: bool = False,
    #
    prepend_initial: bool = False,
    append_terminal: bool = False,
    deviation: bool = False,
    check_singularity: bool = False,
    when_singularity: Literal["critical", "error", "warning", "silent" ] | None = "critical",
    num_variants: int | None = None,
    #
    unpack_singleton: bool = True,
    return_info: bool = False,
) -> Databox | tuple[Databox, _Info]:
    r"""
················································································

==Run Kalman filter on a model using time series data==

Executes a Kalman filter on a model, compliant with `KalmanFilterableProtocol`,
using time series observations from the input Databox. This method enables state
estimation and uncertainty quantification in line with the model's dynamics and
the time series data.

kalman_output = model.kalman_filter(
    input_db,
    span,
    diffuse_scale=None,
    return_=("predict", "update", "smooth", "predict_err", "predict_mse_obs", ),
    return_predict=True,
    return_update=True,
    return_smooth=True,
    return_predict_err=True,
    return_predict_mse_obs=True,
    rescale_variance=False,
    likelihood_contributions=True,
    shocks_from_data=False,
    stds_from_data=False,
    prepend_initial=False,
    append_terminal=False,
    deviation=False,
    check_singularity=False,
    unpack_singleton=True,
    return_info=False,
)

kalman_output, info = model.kalman_filter(
    ...
    return_info=True,
)


### Input arguments ###


???+ input "model"
The model, compliant with `KalmanFilterableProtocol`, performing the
Kalman filtering.

???+ input "input_db"
A Databox containing time series data to be used for filtering.

???+ input "span"
A date span over which the filtering process is executed based on the
measurement time series.

???+ input "diffuse_scale"
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

???+ input "likelihood_contributions"
If `True`, return the contributions of individual periods to the overall
(negative) log likelihood.

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

???+ input "deviation"
If `True`, the constant vectors in transition and measurement equations are
set to zeros, effectively running the Kalman filter as deviations from
steady state (a balanced-growth path)

???+ input "check_singularity"
If `True`, check the one-step ahead MSE matrix for the measurement variables
for singularity, and throw a `SingularMatrixError` exception if the matrix
is singular.

???+ input "unpack_singleton"
If `True`, unpack `out_info` into a plain dictionary for models with a
single variant.

???+ input "return_info"
If `True`, return additional information about the Kalman filtering process.


### Returns ###


???+ returns "kalman_output"
A Databox containing some of the following items (depending on the user requests):

| Attribute         | Type       | Description
|-------------------|---------------------------------------------------
| `predict_med`     | `Databox`  | Medians from the prediction step
| `predict_std`     | `Databox`  | Standard deviations from the prediction step
| `predict_mse_obs` | `list`     | Mean squared error matrices for the prediction step of the available observations of measurement variables
| `update_med`      | `Databox`  | Medians from the update step
| `update_std`      | `Databox`  | Standard deviations from the update step
| `predict_err`     | `Databox`  | Prediction errors
| `smooth_med`      | `Databox`  | Medians from the smoothing step
| `smooth_std`      | `Databox`  | Standard deviations from the smoothing step


???+ returns "out_info"
A dictionary containing additional information about the filtering process,
such as log likelihood and variance scale. For models with multiple
variants, `out_info` is a list of such dictionaries. If
`unpack_singleton=False`, also `out_info` is a one-element list
containing the dictionary for singleton models, too.

················································································
    """
    #[
    work_db = input_db.shallow()
    #if not shocks_from_data:
    #    shock_names = model.get_names(kind=_quantities.ANY_SHOCK, )
    #    work_db.remove(shock_names, strict_names=False, )
    #if not stds_from_data:
    #    std_names = model.get_names(kind=_quantities.ANY_STD, )
    #    work_db.remove(std_names, strict_names=False, )

    num_variants = model.resolve_num_variants_in_context(num_variants, )

    slatable = model.slatable_for_kalman_filter(
        shocks_from_data=shocks_from_data,
        stds_from_data=stds_from_data,
        output_parameters=output_parameters,
    )

    input_ds = Dataslate.from_databox_for_slatable(
        slatable, work_db, span,
        num_variants=num_variants,
        prepend_initial=prepend_initial,
        append_terminal=append_terminal,
        clip_data_to_base_span=True,
    )

    frame = Frame(
        start=input_ds.periods[0],
        end=input_ds.periods[-1],
        simulation_end=input_ds.periods[-1],
    )

    needs = Needs(
        return_=return_,
        return_predict=return_predict,
        return_update=return_update,
        return_smooth=return_smooth,
        return_predict_err=return_predict_err,
        return_predict_mse_obs=return_predict_mse_obs,
        rescale_variance=rescale_variance,
        likelihood_contributions=likelihood_contributions,
    )

    squid = Squid.from_squidable(model, )
    qid_to_logly = model.create_qid_to_logly()

    logly_within_y = tuple(
        i for i, qid in enumerate(squid.y_qids, )
        if qid_to_logly.get(qid, False)
    )

    qid_to_name = model.create_qid_to_name()
    y_names = [ qid_to_name[qid] for qid in squid.y_qids ]
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
    ) if needs.output_store else None

    initialize = _ft.partial(
        _initializers.initialize,
        diffuse_method=diffuse_method,
        diffuse_scale=diffuse_scale,
    )

    #
    # Include the variants even though they are not used because otherwise
    # the number of loops is undetermined (infite iterators)
    main_iter = zip(
        range(num_variants, ),
        model.iter_variants(),
        input_ds.iter_variants(),
    )

    out_info = []

    for vid, model_v, input_ds_v in main_iter:

        variant_header = f"[Variant {vid}]"
        when_singularity = _wd.create_stream(
            when_singularity,
            f"Singularity in prediction MSE matrix in {variant_header}",
        ) if check_singularity else None

        solution_v = model_v._gets_solution(deviation=deviation, )
        data_array = input_ds_v.get_data_variant()

        #
        # Initialize median and MSE of the alpha state vector
        #
        init_cov_u = model_v._gets_cov_unanticipated_shocks()
        initials = initialize(solution_v, init_cov_u, )

        #
        # Presimulate the impact of anticipated shocks
        #
        all_v_impact = (
            _simulate_anticipated_shocks(model_v, input_ds_v, frame, )
            if shocks_from_data else None
        )

        #
        # Get values of observables
        #
        y1_array = data_array[squid.y_qids, :].copy()
        if logly_within_y:
            y1_array[logly_within_y, :] = _np.log(y1_array[logly_within_y, :])
        y1_incidence_array = ~_np.isnan(y1_array)
        #
        #
        output_store_v = _OutputStore(
            input_ds=input_ds_v,
            num_variants=1,
            measurement_names=y_names,
            needs=needs,
            name_to_log_name=name_to_log_name,
            transform=solution_v.Ua,
            squid=squid,
            measurement_incidence=y1_incidence_array,
        ) if needs.output_store else None

        partial_generate_period_system = _ft.partial(
            generate_period_system,
            solution_v=solution_v,
            y1_array=y1_array,
            std_u_array=data_array[squid.std_u_qids, :],
            std_w_array=data_array[squid.std_w_qids, :],
            all_v_impact=all_v_impact,
        )

        partial_generate_period_data = _ft.partial(
            generate_period_data,
            y_array=y1_array,
            u_array=data_array[squid.u_qids, :],
            v_array=data_array[squid.v_qids, :],
            w_array=data_array[squid.w_qids, :],
        )

        store_predict = (
            _ft.partial(_OutputStore.store_predict, self=output_store_v, )
            if needs.return_predict else None
        )

        store_update = (
            _ft.partial(_OutputStore.store_update, self=output_store_v, )
            if needs.return_update else None
        )

        store_smooth = (
            _ft.partial(_OutputStore.store_smooth, self=output_store_v, )
            if needs.return_smooth else None
        )

        cache = predict(
            num_periods=input_ds_v.num_periods,
            initials=initials,
            partial_generate_period_system=partial_generate_period_system,
            partial_generate_period_data=partial_generate_period_data,
            store_predict=store_predict,
            store_update=store_update,
            store_smooth=store_smooth,
            check_singularity=check_singularity,
            when_singularity=when_singularity,
        )

        if when_singularity is not None:
            when_singularity._raise()

        if cache.needs_estimate_unknown_init:
            estimate_unknown_init(
                cache=cache,
            )
            correct_for_unknown_init(
                cache=cache,
                store_predict=store_predict,
            )

        if needs.return_update:
            update(cache=cache, store_update=store_update, )

        if needs.return_smooth:
            smooth(cache=cache, store_smooth=store_smooth, )

        cache.calculate_likelihood(rescale_variance=needs.rescale_variance, )
        if needs.likelihood_contributions:
            cache.calculate_likelihood_contributions()

        if needs.output_store:
            output_store_v.rescale_stds(cache.var_scale, )

        out_info_v = cache.create_out_info(span, )

        if needs.output_store:
            output_store.extend(output_store_v, )

        out_info.append(out_info_v, )

    out_data = (
        output_store.create_out_data()
        if output_store is not None else None
    )
    #
    if return_info:
        out_info = _has_variants.unpack_singleton(
            out_info, model.is_singleton,
            unpack_singleton=True,
        )
        return out_data, out_info,
    else:
        return out_data
    #]


class Cache:
    """
    """
    #[

    _LOG_2_PI = _np.log(2 * _np.pi)

    _slots_to_input = (
        "num_periods",
    )

    _slots_to_preallocate = (
        "all_y",
        "all_num_obs",
        "all_Fi",
        "all_L",
        "all_a0",
        "all_y0",
        "all_u0",
        "all_v0",
        "all_w0",
        "all_Q0",
        "all_pe",
        "all_Zt_Fi",
        "all_T_G_prev",
        "all_H_cov_w",
        "all_P_cov_u",
        "all_Xi",
        "all_Z",
        "all_G",
        "all_M",
    )

    _other_slots = (
        "needs_estimate_unknown_init",
        "unknown_init_estimate",
        "sum_num_obs",
        "sum_log_det_F",
        "sum_pe_Fi_pe",
        "var_scale",
        "neg_log_likelihood",
        "neg_log_likelihood_contributions",
        "all_log_det_F",
        "all_pe_Fi_pe",
    )

    # __slots__ = _slots_to_input + _slots_to_preallocate + _other_slots

    def __init__(
        self: Self,
        **kwargs,
    ) -> None:
        """
        """
        for n in self._slots_to_input:
            setattr(self, n, kwargs[n], )
        for n in self._slots_to_preallocate:
            setattr(self, n, [None] * self.num_periods, )
        for n in self._other_slots:
            setattr(self, n, None, )

    def calculate_likelihood(self: Self, rescale_variance: bool, ) -> None:
        """
        """
        self.all_pe_Fi_pe = tuple(
            pe @ Fi @ pe if pe is not None and Fi is not None else None
            for pe, Fi, in zip(self.all_pe, self.all_Fi, )
        )
        all_det_Fi = tuple(
            float(_np.linalg.det(Fi)) if Fi is not None else None
            for Fi in self.all_Fi
        )
        self.all_log_det_F = tuple(
            float(-_np.log(det_Fi)) if det_Fi is not None else None
            for det_Fi in all_det_Fi
        )
        self.sum_num_obs = sum(i for i in self.all_num_obs if i is not None)
        self.sum_log_det_F = sum(i for i in self.all_log_det_F if i is not None)
        self.sum_pe_Fi_pe = sum(i for i in self.all_pe_Fi_pe if i is not None)
        self.var_scale = 1
        #
        if rescale_variance:
            self._calculate_variance_scale()
        #
        self.neg_log_likelihood = (
            + self.sum_num_obs*self._LOG_2_PI
            + self.sum_log_det_F
            + self.sum_pe_Fi_pe
        ) / 2;

    def calculate_likelihood_contributions(self: Self, ) -> None:
        """
        """
        self.neg_log_likelihood_contributions = tuple(
            (log_det_F + pe_Fi_pe + num_obs * self._LOG_2_PI)/2 if num_obs else 0
            for log_det_F, pe_Fi_pe, num_obs in zip(
                self.all_log_det_F, self.all_pe_Fi_pe, self.all_num_obs,
            )
            # if num_obs is not None
        )

    def _calculate_variance_scale(self: Self, /, ) -> None:
        """
        """
        if self.sum_num_obs == 0:
            self.var_scale = 1
            self.sum_pe_Fi_pe = 0
            return
        self.var_scale = self.sum_pe_Fi_pe / self.sum_num_obs
        self.sum_log_det_F += self.sum_num_obs * _np.log(self.var_scale)
        self.sum_pe_Fi_pe = self.sum_pe_Fi_pe / self.var_scale

    def create_out_info(self: Self, span: Iterable[Period], ) -> dict[str, Any]:
        """
        """
        out_info = {
            "neg_log_likelihood": float(self.neg_log_likelihood),
            "log_det_F": Series(periods=span, values=self.all_log_det_F, ),
            "var_scale": float(self.var_scale),
            "std_scale": float(_covariances.sqrt_positive(self.var_scale)),
        }
        if self.neg_log_likelihood_contributions is not None:
            out_info["neg_log_likelihood_contributions"] = Series(
                periods=span,
                values=self.neg_log_likelihood_contributions,
            )
        return out_info

    #]


def predict(
    num_periods: int,
    initials: tuple[_np.ndarray, _np.ndarray, _np.ndarray | None],
    partial_generate_period_system: Callable,
    partial_generate_period_data: Callable,
    store_predict: Callable | None = None,
    store_update: Callable | None = None,
    store_smooth: Callable | None = None,
    check_singularity: bool = False,
    when_singularity: _wd.Stream | None = None,
) -> None:
    """
    """
    #[
    cache = Cache(num_periods=num_periods, )
    a1_prev, Q1_prev, Xi_prev, = initials
    needs_estimate_unknown_init = Xi_prev is not None
    cache.needs_estimate_unknown_init = needs_estimate_unknown_init
    cache.last_period_of_observations = -1
    create_empty = _ft.partial(_np.empty, shape=(0, 0), dtype=_np.float64, )
    #
    for t in range(num_periods, ):
        #
        T, P, K, Z, H, D, cov_u, cov_w, v_impact, U, *_ = partial_generate_period_system(t, )
        y1, u0, v0, w0, inx_y, *_ = partial_generate_period_data(t, )
        #
        cache.all_y[t] = y1
        cache.all_num_obs[t] = y1.size
        any_y = cache.all_num_obs[t] > 0
        if any_y:
            cache.last_period_of_observations = t
        #
        if t > 0 and store_smooth:
            T_G_prev = T @ G_prev
            cache.all_T_G_prev[t-1] = T_G_prev
            cache.all_L[t-1] = T - T_G_prev @ Z_prev
        #
        # MSE prediction step
        #
        if P is not None:
            P_u0 = P @ u0
            P_cov_u = P @ cov_u
            P_cov_u_Pt = P_cov_u @ P.T
        else:
            P_u0 = u0
            P_cov_u = cov_u
            P_cov_u_Pt = cov_u
        #
        Q0 = T @ Q1_prev @ T.T + P_cov_u_Pt
        Q0 = _covariances.symmetrize(Q0)
        H_cov_w = H @ cov_w
        #
        F = (
            Z @ Q0 @ Z.T + H_cov_w @ H.T if any_y
            else create_empty()
        )
        F = _covariances.symmetrize(F, )
        #
        # sing_values = _np.linalg.svd(F, compute_uv=False, )
        # relative_cond = sing_values[-1] / sing_values[0]
        # print(t, F.size, relative_cond, sing_values[0], )
        # TODO: Check if F singularity if requested by user
        inv = _INVERSE_FUNCTION["regular"]
        if when_singularity and any_y:
            inv = _check_singularity(F, t, inx_y, when_singularity, )

        Fi = inv(F) if any_y else create_empty()
        Fi = _covariances.symmetrize(Fi, )
        #
        # Median prediction step
        #
        a0 = T @ a1_prev + K + P_u0 + (v_impact if v_impact is not None else 0)
        y0 = Z @ a0 + D + H @ w0
        #
        if store_predict:
            store_predict(t=t, a0=a0, y0=y0, u0=u0, v0=v0, w0=w0, Q0=Q0, F=F, cov_u0=cov_u, cov_w0=cov_w, )
        #
        # MSE updating step
        #
        Zt_Fi = Z.T @ Fi
        G = Q0 @ Zt_Fi
        Q1 = Q0 - G @ Z @ Q0
        Q1 = _covariances.symmetrize(Q1)
        if store_update:
            store_update(t=t, Q=Q1, )
        #
        # Median updating step
        #
        pe = y1 - y0
        a1 = a0 + G @ pe
        # x1 = U @ a1
        #
        cache.all_a0[t] = a0
        cache.all_y0[t] = y0
        cache.all_pe[t] = pe
        cache.all_Fi[t] = Fi
        cache.all_Z[t] = Z
        #
        if store_smooth:
            cache.all_u0[t] = u0
            cache.all_v0[t] = v0
            cache.all_w0[t] = w0
            cache.all_Q0[t] = Q0
            cache.all_Zt_Fi[t] = Zt_Fi
            cache.all_P_cov_u[t] = P_cov_u
            cache.all_H_cov_w[t] = H_cov_w
        #
        if needs_estimate_unknown_init:
            if t == 0:
                Xi = T @ Xi_prev
            else:
                Xi = (T - T @ G_prev @ Z_prev) @ Xi_prev
            cache.all_Xi[t] = Xi
            cache.all_G[t] = G
            Xi_prev = Xi
        #
        a1_prev = a1
        Q1_prev = Q1
        G_prev = G
        Z_prev = Z
    #
    return cache
    #]


def estimate_unknown_init(
    cache: Cache,
    delta_tolerance: float = _DEFAULT_DELTA_TOLERANCE,
) -> None:
    """
    """
    #[
    all_M = tuple(
        (Z @ Xi) if Z is not None and Xi is not None else None
        for Z, Xi in zip(cache.all_Z, cache.all_Xi, )
    )
    all_Mt_Fi = tuple(
        (M.T @ Fi) if M is not None and Fi is not None else None
        for M, Fi in zip(all_M, cache.all_Fi, )
    )
    sum_Mt_Fi_M = sum(
        (Mt_Fi @ M) if Mt_Fi is not None and M is not None else 0
        for Mt_Fi, M in zip(all_Mt_Fi, all_M, )
    )
    sum_Mt_Fi_pe = sum(
        (Mt_Fi @ pe) if Mt_Fi is not None and pe is not None else 0
        for Mt_Fi, pe in zip(all_Mt_Fi, cache.all_pe, )
    )
    #
    sum_Mt_Fi_M = _covariances.symmetrize(sum_Mt_Fi_M)
    sum_Mt_Fi_M[_np.abs(sum_Mt_Fi_M) < delta_tolerance] = 0
    #
    delta, *_ = _np.linalg.lstsq(sum_Mt_Fi_M, sum_Mt_Fi_pe, rcond=None, )
    #
    cache.unknown_init_estimate = delta
    cache.all_M = all_M
    #]


def correct_for_unknown_init(
    cache: Cache,
    store_predict: Callable | None,
) -> None:
    """
    """
    #[
    delta = cache.unknown_init_estimate
    for t in range(cache.num_periods, ):
        cache.all_a0[t] += cache.all_Xi[t] @ delta
        M_delta = cache.all_M[t] @ delta
        cache.all_y0[t] += M_delta
        cache.all_pe[t] -= M_delta
        if store_predict:
            store_predict(t=t, a0=cache.all_a0[t], y0=cache.all_y0[t], )
    #]


def update(
    cache: Cache,
    store_update: Callable | None,
) -> None:
    """
    """
    #[
    for t in range(cache.num_periods, ):
        a1, y1, u1, v1, w1, Q1, *_ = one_step_back(t, cache, )
        if store_update:
            pe = cache.all_pe[t]
            store_update(t=t, xi=a1, y=y1, u=u1, v=v1, w=w1, pe=pe, )
    #]


def smooth(
    cache: Cache,
    store_smooth: Callable | None,
) -> None:
    """
    """
    #[
    N = None
    r = None
    for t in reversed(range(cache.num_periods, )):
        a2, y2, u2, v2, w2, Q2, N, r = one_step_back(t, cache, N, r, )
        if store_smooth:
            store_smooth(t=t, xi=a2, y=y2, u=u2, v=v2, w=w2, Q=Q2, )
    #]


def one_step_back(
    t: int,
    cache: Cache,
    N: _np.ndarray | None = None,
    r: _np.ndarray | None = None,
    /,
) -> tuple:
    """
    """
    #[
    ak = cache.all_a0[t]
    y = cache.all_y[t]
    uk = cache.all_u0[t]
    vk = cache.all_v0[t]
    wk = cache.all_w0[t]
    Qk = cache.all_Q0[t]
    #
    if t <= cache.last_period_of_observations:
        T_G_prev = cache.all_T_G_prev[t]
        P_cov_u = cache.all_P_cov_u[t]
        H_cov_w = cache.all_H_cov_w[t]
        #
        a0 = cache.all_a0[t]
        Q0 = cache.all_Q0[t]
        L = cache.all_L[t]
        Zt_Fi = cache.all_Zt_Fi[t]
        Fi = cache.all_Fi[t]
        Z = cache.all_Z[t]
        pe = cache.all_pe[t]
        #
        Fi_pe = Fi @ pe
        Zt_Fi_pe = Zt_Fi @ pe
        Zt_Fi_Z = Zt_Fi @ Z
        #
        if N is None:
            N = Zt_Fi_Z
        else:
            N = Zt_Fi_Z + L.T @ N @ L
        Qk = Qk - Q0 @ N @ Q0
        Qk = _covariances.symmetrize(Qk)
        #
        if r is None:
            wk = wk + H_cov_w.T @ Fi_pe
            r = Zt_Fi_pe
        else:
            wk = wk + H_cov_w.T @ (Fi_pe - (T_G_prev.T @ r))
            r = Zt_Fi_pe + L.T @ r
        ak = a0 + Q0 @ r
        uk = uk + P_cov_u.T @ r
        vk = vk
    #
    return ak, y, uk, vk, wk, Qk, N, r
    #]


_INVERSE_FUNCTION = {
    "regular": _np.linalg.inv,
    "singular": _np.linalg.pinv,
}


def _check_singularity(
    F: _np.ndarray,
    t: int,
    boolex_y: list[bool],
    when_singularity: _wd.Stream | None,
) -> Callable | NoReturn:
    """
    """
    rank_F = _np.linalg.matrix_rank(F, )
    if rank_F == F.shape[0]:
        return _INVERSE_FUNCTION["regular"]
    index_y = _np.where(boolex_y, )[0].tolist()
    message = f"Singular prediction MSE matrix in period {t}, rank {rank_F} of {F.shape[0]}"
    # u, s, v, = _np.linalg.svd(F, )
    when_singularity.add(message, )
    return _INVERSE_FUNCTION["singular"]

