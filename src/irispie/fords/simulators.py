"""
First-order system simulators
==============================

Table of contents
------------------

* `simulate`
* `simulate_flat`
* `simulate_conditional`
* `simulate_measurement`
* `_generate_period_system`
* `_store_smooth`
* `get_init_xi`

"""


#[
from __future__ import annotations

from typing import (Protocol, )
import numpy as _np
import functools as _ft
import wlogging as _wl

from ..dates import (Period, )
from ..frames import (Frame, )
from ..dataslates.main import (Dataslate, )
from ..plans.simulation_plans import (SimulationPlan, )
from .. import frames as _frames
from ..incidences import main as _incidences
from ..incidences.main import (Token, )
from .solutions import (Solution, )
from .descriptors import (SolutionVectors, Squid, )
from .kalmans import (Cache, )
from . import kalmans as _kalmans
from . import shock_simulators as _shock_simulators

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, )
    from collections.abc import (Iterable, )
#]


_extract_shock_values \
    = _shock_simulators.extract_shock_values


_simulate_anticipated_shocks \
    = _shock_simulators.simulate_square_anticipated_shocks


class FordSimulatableProtocol(Protocol, ):
    """
    """
    #[

    num_variants: int
    solution_vectors: SolutionVectors
    def get_singleton_solution(self, deviation: bool, ) -> Solution: ...

    #]


def simulate(
    model_v: FordSimulatableProtocol,
    dataslate_v: Dataslate,
    plan: SimulationPlan | None,
    vid: int,
    *,
    logger: _wl.Logger,
    #
    deviation: bool = False,
    check_singularity: bool = False,
    force_split_frames: bool = False,
    return_info: bool = False,
) -> dict[str, Any]:
    """
    """
    #[
    periods = dataslate_v.periods
    num_periods = dataslate_v.num_periods
    data_array = dataslate_v.get_data_variant(0, )
    squid = Squid(model_v, )
    #
    # In first-order simulation, splitting into frames is necessary only if
    # there are some endogenized anticipated shocks and, at the same time, some
    # nonzero unanticipated shocks (not counting the first period in either case)
    #
    is_plan_empty = plan is None or plan.is_empty
    needs_splitting = _check_needs_splitting(
        plan=plan,
        dataslate=dataslate_v,
        squid=squid,
        is_plan_empty=is_plan_empty,
        force_split_frames=force_split_frames,
    )
    logger.debug(f"Needs splitting: {needs_splitting}")
    #
    if needs_splitting:
        create_frames = _frames.split_into_frames
    else:
        create_frames = _frames.setup_single_frame
    #
    if is_plan_empty:
        simulate_frame = simulate_flat
    else:
        plan_registers = _get_plan_registers_as_bool_arrays(plan, periods, )
        simulate_frame = _ft.partial(
            simulate_conditional,
            check_singularity=check_singularity,
            input_data_array=data_array.copy(),
            plan_registers=plan_registers,
            Z_xi=_create_Z_xi(squid, )
        )
    #
    # Split the base span into frames
    # Each frame is simulated until the end of the dataslate base span
    #
    # TODO: Optimize the simulation end of each frame based on the presence of
    # endogenized anticipated shocks and nonzero anticipated shocks
    #
    dataslate_end = dataslate_v.base_periods[-1]
    frames = create_frames(
        model_v, dataslate_v, plan,
        get_simulation_end=lambda *_, : dataslate_end,
    )
    logger.debug(f"Splitting simulation into {len(frames)} frame(s)")
    #
    # Resolve columns within each frame based on the start date of the dataslate
    for frame in frames:
        frame.resolve_columns(periods[0], )
    #
    #
    #===========================================================================
    xi_array = _np.full((squid.num_xi, num_periods, ), _np.nan, )
    frame_db = ()
    for frame in frames:
        #
        if needs_splitting:
            frame_ds = dataslate_v.copy(invariant=False, )
            frame.reset_unanticipated(frame_ds, model_v, )
        else:
            frame_ds = dataslate_v
        #
        xi_array_in_frame = simulate_frame(
            model_v, frame_ds, frame, vid,
            logger=logger,
            deviation=deviation,
        )
        xi_array[:, frame.slice] = xi_array_in_frame[:, frame.slice]
        #
        simulate_measurement(
            model_v, dataslate_v, frame, xi_array, vid,
            logger=logger,
            deviation=deviation,
        )
        #
        # Copy frame data back to the main dataslate if there are more than one
        # frame; otherwise the main dataslate is the same as the frame dataslate
        if needs_splitting:
            frame.copy_frame_data(dataslate_v, frame_ds, )
        #
        # Create a frame databox only if output info is requested
        if return_info:
            frame_db = frame_db + (frame_ds.to_databox(), )
    #===========================================================================
    #
    #
    info = {
        "method": "first_order",
        "frames": frames,
        "frame_db": frame_db,
    }
    #
    return info
    #]


def simulate_flat(
    model_v: FordSimulatableProtocol,
    dataslate_v: Dataslate,
    frame: Frame,
    vid: int,
    *,
    logger: _wl.Logger,
    deviation: bool,
) -> _np.ndarray:
    """
    """
    #[
    logger.debug(f"Running flat simulation {frame.start}>>{frame.end}>>{frame.simulation_end}")
    solution = model_v.get_singleton_solution(deviation=deviation, )
    vec = model_v.solution_vectors
    num_periods = dataslate_v.num_periods
    simulation_columns = tuple(range(*frame.simulation_slice.indices(num_periods, ), ), )
    curr_xi_qids, curr_xi_indexes = vec.get_curr_transition_indexes()
    #
    T = solution.T
    P = solution.P
    K = solution.K
    #
    dataslate_v.logarithmize()
    data_array = dataslate_v.get_data_variant(0, )
    num_xi = len(vec.transition_variables)
    xi_array = _np.full((num_xi, num_periods, ), _np.nan, )
    u_array = _extract_shock_values(data_array, vec.unanticipated_shocks, )
    all_v_impact = _simulate_anticipated_shocks(model_v, dataslate_v, frame, )
    #
    # Simulate one period at a time
    # Store results in data_array
    # Capture xi_array for later use
    #
    xi = get_init_xi(data_array, vec.transition_variables, simulation_columns[0], )
    zero_false_init_xi(xi, vec.true_initials, )
    for t in simulation_columns:
        xi = T @ xi + P @ u_array[:, t] + K
        if all_v_impact[t] is not None:
            xi += all_v_impact[t]
        xi_array[:, t] = xi
        data_array[curr_xi_qids, t] = xi[curr_xi_indexes, ...]
    #
    dataslate_v.delogarithmize()
    #
    return xi_array
    #]


def simulate_conditional(
    model_v: FordSimulatableProtocol,
    dataslate_v: Dataslate,
    frame: Frame,
    vid: int,
    *,
    logger: _wl.Logger,
    deviation: bool,
    check_singularity: bool,
    #
    plan_registers: dict[str, _np.ndarray],
    input_data_array: _np.ndarray,
    Z_xi: _np.ndarray,
) -> _np.ndarray:
    """
    """
    #[
    logger.debug(f"Running conditional simulation {frame.start}>>{frame.end}>>{frame.simulation_end}")
    solution = model_v.get_singleton_solution(deviation=deviation, )
    vec = model_v.solution_vectors
    periods = dataslate_v.periods
    num_periods = dataslate_v.num_periods
    squid = Squid(model_v, )
    #
    dataslate_v.logarithmize()
    data_array = dataslate_v.get_data_variant()
    #
    # Prepare array of all xi values (needed for measurement simulation)
    xi_array = _np.full((squid.num_xi, num_periods, ), _np.nan, )
    #
    curr_xi_qids, curr_xi_indexes = vec.get_curr_transition_indexes()
    #
    # Initialize Kalman filter:
    # [initial median, initial MSE, number of initials to estimates, ]
    init_med = get_init_xi(data_array, vec.transition_variables, frame.first, )
    zero_false_init_xi(init_med, vec.true_initials, )
    init_mse = _np.zeros((squid.num_xi, squid.num_xi), )
    unknown_init_impact = None
    initials = (init_med, init_mse, unknown_init_impact, )
    #
    u0_array = data_array[squid.u_qids, frame.simulation_slice]
    v0_array = data_array[squid.v_qids, frame.simulation_slice]
    w0_array = data_array[squid.w_qids, frame.simulation_slice]
    #
    if model_v.is_deterministic:
        std_u_array = _np.ones((squid.num_u, frame.num_simulation_columns, ), )
        std_v_array = _np.ones((squid.num_v, frame.num_simulation_columns, ), )
        std_w_array = _np.ones((squid.num_w, frame.num_simulation_columns, ), )
    else:
        std_u_array = data_array[squid.std_u_qids, frame.simulation_slice]
        std_v_array = data_array[squid.std_v_qids, frame.simulation_slice]
        std_w_array = data_array[squid.std_w_qids, frame.simulation_slice]
    #
    all_v_impact = _simulate_anticipated_shocks(model_v, dataslate_v, frame, )
    #
    # Preallocate default data
    curr_xi_exogenized = _np.full((squid.num_curr_xi, num_periods, ), _np.nan, )
    std_u_endogenized = _np.zeros((squid.num_u, num_periods, ), )
    std_w_endogenized = _np.zeros((squid.num_w, num_periods, ), )
    #
    # Insert SimulationPlan information
    incidence_v, forward, Rx, std_v_endogenized = None, None, None, None
    if plan_registers:
        packed = (plan_registers, input_data_array, squid, frame.slice, logger, )
        _insert_exogenized_unanticipated(curr_xi_exogenized, packed, )
        _insert_exogenized_anticipated(curr_xi_exogenized, packed, )
        _insert_endogenized_unanticipated(std_u_endogenized, packed, )
        #
        incidence_v = plan_registers["endogenized_anticipated"][:, frame.simulation_slice]
        sum_incidence_v = incidence_v.sum()
        incidence_v = incidence_v if sum_incidence_v else None
        if incidence_v is not None:
            forward = incidence_v.any(axis=0, ).nonzero()[0].max()
            Rx = solution.expand_square_solution(forward, )
            std_v_endogenized = std_v_array[incidence_v, ]
        logger.debug(f"Endogenized anticipated: {sum_incidence_v}")
    #
    curr_xi_exogenized = curr_xi_exogenized[:, frame.simulation_slice]
    std_u_endogenized = std_u_endogenized[:, frame.simulation_slice]
    std_w_endogenized = std_w_endogenized[:, frame.simulation_slice]
    #
    # Define the Kalman callbacks
    generate_period_system = _ft.partial(
        _generate_period_system,
        initials=initials,
        solution=solution,
        Z_xi=Z_xi,
        curr_xi_exogenized=curr_xi_exogenized,
        std_u_endogenized=std_u_endogenized,
        std_w_endogenized=std_w_endogenized,
        std_v_endogenized=std_v_endogenized,
        all_v_impact=all_v_impact[frame.simulation_slice],
        incidence_v=incidence_v,
        Rx=Rx,
    )
    #
    generate_period_data = _ft.partial(
        _generate_period_data,
        curr_xi_exogenized=curr_xi_exogenized,
        u_array=u0_array,
        v_array=v0_array,
        w_array=w0_array,
    )
    store_smooth = _ft.partial(
        _store_smooth,
        data_array=data_array[:, frame.simulation_slice],
        xi_array=xi_array[:, frame.simulation_slice],
        incidence_v=incidence_v,
        squid=squid,
    )
    #
    # Run Kalman filter and smoother
    cache = _kalmans.predict(
        num_periods=frame.num_simulation_columns,
        generate_period_system=generate_period_system,
        generate_period_data=generate_period_data,
        store_smooth=True,
        check_singularity=check_singularity,
    )
    _kalmans.smooth(
        cache=cache,
        store_smooth=store_smooth,
    )
    dataslate_v.delogarithmize()
    #
    return xi_array
    #]


def simulate_measurement(
    model_v: FordSimulatableProtocol,
    dataslate_v: Dataslate,
    frame: Frame,
    xi_array: _np.ndarray,
    vid: int,
    *,
    logger: _wl.Logger,
    deviation: bool = False,
    **kwargs,
) -> None:
    """
    """
    #[
    logger.debug("Running measurement simulation")
    solution = model_v.get_singleton_solution(deviation=deviation, )
    vec = model_v.solution_vectors
    num_periods = dataslate_v.num_periods
    #
    dataslate_v.logarithmize()
    data_array = dataslate_v.get_data_variant(0, )
    #
    Z = solution.Z
    H = solution.H
    D = solution.D if not deviation else 0
    #
    y_qids = [t.qid for t in vec.measurement_variables]
    w_array = _extract_shock_values(data_array, vec.measurement_shocks, )
    #
    for t in range(*frame.simulation_slice.indices(num_periods), ):
        data_array[y_qids, t] = Z @ xi_array[:, t] + H @ w_array[:, t] + D
    #
    dataslate_v.delogarithmize()
    #]


def _generate_period_system(
    t: int,
    #
    initials: list[_np.ndarray, _np.ndarray, int, ],
    solution: Solution,
    Z_xi: _np.ndarray | None,
    curr_xi_exogenized: _np.ndarray | None,
    std_u_endogenized: _np.ndarray | None,
    std_w_endogenized: _np.ndarray | None,
    std_v_endogenized: _np.ndarray | None,
    all_v_impact: Iterable[_np.ndarray | None],
    incidence_v: _np.ndarray | None,
    Rx: list[_np.ndarray] | None,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    #
    # Initial conditions
    if t == -1:
        return _generate_initials(initials, std_v_endogenized, )
    #
    # Period t system matrices
    T = solution.T
    P = solution.P
    K = solution.K
    Z = _generate_Z(t, Z_xi, curr_xi_exogenized, )
    num_xi = T.shape[0]
    num_y = Z.shape[0]
    H = _np.zeros((num_y, solution.num_w, ), )
    D = _np.zeros((num_y, ), )
    cov_u = _np.diag(std_u_endogenized[:, t]**2, )
    cov_w = _np.diag(std_w_endogenized[:, t]**2, )
    v_impact = all_v_impact[t]
    #
    if incidence_v is not None and incidence_v.any():
        #
        # Endogenized anticipated shocks
        R = _generate_R(t, incidence_v, Rx, num_xi, )
        num_v_endogenized = R.shape[1]
        T = _np.block([
            [T, R],
            [_np.zeros((num_v_endogenized, num_xi)), _np.eye(num_v_endogenized)],
        ])
        P = _np.pad(P, ((0, num_v_endogenized), (0, 0)), )
        K = _np.pad(K, (0, num_v_endogenized), )
        Z = _np.pad(Z, ((0, 0), (0, num_v_endogenized)), )
        if v_impact is not None:
            v_impact = _np.pad(v_impact, (0, num_v_endogenized), )
    #
    return T, P, K, Z, H, D, cov_u, cov_w, v_impact,
    #]


def _generate_initials(
    initials: list[_np.ndarray, _np.ndarray, int, ],
    std_v_endogenized: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    """
    """
    #[
    num_v_endogenized = (
        len(std_v_endogenized)
        if std_v_endogenized is not None else 0
    )
    #
    # Extend initial conditions to include endogenized anticipated shocks
    if num_v_endogenized > 0:
        init_med = initials[0]
        init_mse = initials[1]
        uknown_init_impact = None
        initials = (
            _np.pad(initials[0], (0, num_v_endogenized), ),
            _block_diag(initials[1], _np.diag(std_v_endogenized**2), ),
            uknown_init_impact,
        )
    #
    # Return updated initial conditions
    return initials
    #]


def _generate_Z(
    t: int,
    Z_xi: _np.ndarray | None,
    curr_xi_exogenized: _np.ndarray | None,
) -> _np.ndarray:
    """
    """
    #[
    if curr_xi_exogenized is not None:
        inx_y = ~_np.isnan(curr_xi_exogenized[:, t])
        return Z_xi[inx_y, :]
    else:
        return _np.zeros((0, Z_xi.shape[1], ), )
    #]


def _generate_R(
    t: int,
    incidence_v: _np.ndarray,
    Rx: list[_np.ndarray],
    num_xi: int,
) -> _np.ndarray:
    """
    """
    #[
    def zeros(i):
        return _np.zeros((num_xi, i.sum()), )
    Rx_t = [None]*t + Rx[:len(Rx)-t]
    return _np.hstack(tuple(
        r[:, i] if r is not None else zeros(i)
        for r, i in zip(Rx_t, incidence_v.T, )
    ))
    #]


def _generate_period_data(
    t: int,

    curr_xi_exogenized: _np.ndarray | None,
    u_array: _np.ndarray,
    v_array: _np.ndarray,
    w_array: _np.ndarray,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    if curr_xi_exogenized is not None:
        inx_y = ~_np.isnan(curr_xi_exogenized[:, t], )
        y = curr_xi_exogenized[inx_y, t]
    else:
        y = _np.zeros((0, ), )
    u = u_array[:, t]
    v = v_array[:, t]
    w = w_array[:, t]
    return y, u, v, w,
    #]


def _store_smooth(
    data_array: _np.ndarray,
    xi_array: _np.ndarray,
    incidence_v: _np.ndarray | None,
    squid: Squid,
    #
    t: int,
    xi: _np.ndarray | None = None,
    u: _np.ndarray | None = None,
    v: _np.ndarray | None = None,
    w: _np.ndarray | None = None,
    **kwargs,
) -> None:
    """
    """
    #[
    def update_v_endogenized(v_endogenized, ):
        ___ = data_array[squid.v_qids, :]
        ___[incidence_v] += v_endogenized
        data_array[squid.v_qids, :] = ___
    #
    if xi is not None:
        if incidence_v is not None:
            num_v_endogenized = incidence_v.sum()
            v_endogenized = xi[-num_v_endogenized:]
            xi = xi[:-num_v_endogenized]
            if t == data_array.shape[1] - 1:
                update_v_endogenized(v_endogenized, )
        data_array[squid.curr_xi_qids, t] = xi[squid.curr_xi_indexes, ...]
        xi_array[:, t] = xi
    if u is not None:
        data_array[squid.u_qids, t] = u
    if w is not None:
        data_array[squid.w_qids, t] = w
    #]


def get_init_xi(
    maybelog_working_data: _np.ndarray,
    transition_solution_vector: Iterable[Token, ...],
    first_column: int,
) -> _np.ndarray:
    """
    """
    #[
    init_xi_rows, init_xi_columns = _incidences.rows_and_columns_from_tokens(
        transition_solution_vector,
        first_column - 1,
    )
    return maybelog_working_data[init_xi_rows, init_xi_columns]
    #]


def zero_false_init_xi(
    init_xi: _np.ndarray,
    true_initials: Iterable[bool, ...],
) -> None:
    #[
    false_initials = [ (not i) for i in true_initials ]
    init_xi[false_initials, ...] = 0
    #]


def _get_plan_registers_as_bool_arrays(
    plan: SimulationPlan,
    periods: tuple[Period, ...],
) -> dict[str, _np.ndarray]:
    """
    """
    #[
    register_names = (
        "exogenized_anticipated",
        "endogenized_anticipated",
        "exogenized_unanticipated",
        "endogenized_unanticipated",
    )
    get_register = _ft.partial(
        plan.get_register_as_bool_array,
        names=...,
        periods=periods,
    )
    return {
        n: get_register(register_name=n, )
        for n in register_names
    }
    #]


def _create_Z_xi(squid, /, ) -> _np.ndarray:
    """
    """
    #[
    Z_xi = _np.zeros((squid.num_curr_xi, squid.num_xi, ), )
    Z_xi[range(squid.num_curr_xi, ), squid.curr_xi_indexes] = 1
    return Z_xi
    #]


def _insert_exogenized_unanticipated(
    curr_xi_exogenized: _np.ndarray,
    packed: tuple,
) -> None:
    """
    """
    #[
    plan_registers, input_data_array, squid, frame_slice, logger = packed
    #
    ___ = plan_registers["exogenized_unanticipated"][:, frame_slice]
    curr_xi_exogenized[:, frame_slice][___] \
        = input_data_array[squid.curr_xi_qids, frame_slice][___]
    #
    logger.debug(f"Exogenized unanticipated: {___.sum()}")
    #]


def _insert_exogenized_anticipated(
    curr_xi_exogenized: _np.ndarray,
    packed: tuple,
) -> None:
    """
    """
    #[
    plan_registers, input_data_array, squid, frame_slice, logger = packed
    #
    ___ = plan_registers["exogenized_anticipated"][:, frame_slice]
    curr_xi_exogenized[:, frame_slice][___] \
        = input_data_array[squid.curr_xi_qids, frame_slice][___]
    #
    logger.debug(f"Exogenized anticipated: {___.sum()}")
    #]


def _insert_endogenized_unanticipated(
    std_u_endogenized: _np.ndarray,
    packed: tuple,
) -> None:
    """
    """
    #[
    plan_registers, input_data_array, squid, frame_slice, logger = packed
    #
    ___ = plan_registers["endogenized_unanticipated"][:, frame_slice]
    std_u_endogenized[:, frame_slice][___] \
        = input_data_array[squid.std_u_qids, frame_slice][___]
    #
    logger.debug(f"Endogenized unanticipated: {___.sum()}")
    #]



def _block_diag(A, B):
    """
    Concatenate two square matrices into a block diagonal matrix
    """
    #[
    return _np.block([
        [A, _np.zeros((A.shape[0], B.shape[1]), )],
        [_np.zeros((B.shape[0], A.shape[1]), ), B],
    ])
    #]


def _check_needs_splitting(
    plan: SimulationPlan | None,
    dataslate: Dataslate,
    squid: Squid,
    is_plan_empty: bool,
    force_split_frames: bool,
) -> bool:
    """
    """
    #[
    if force_split_frames:
        return True
    #
    if is_plan_empty and not force_split_frames:
        return False
    #
    any_endogenized_anticipated = (
        plan is not None
        and plan.any_endogenized_anticipated_except_start
    )
    if not any_endogenized_anticipated:
        return False
    #
    base_columns_except_start = dataslate.base_columns[1:]
    data_array = dataslate.get_data_variant(0, )
    any_nonzero_unanticipated_shocks \
            = data_array \
            [squid.v_qids, :] \
            [:, base_columns_except_start] \
            .any()
    if not any_nonzero_unanticipated_shocks:
        return False
    #
    return True
    #]


