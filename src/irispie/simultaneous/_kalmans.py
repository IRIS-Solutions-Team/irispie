r"""
Kalman filter inlay
"""


#[

from __future__ import annotations

import numpy as _np
import documark as _dm

from ..fords import kalmans as _kalmans

#]


# TODO: Create Kalmanable protocol


class Inlay:
    r"""
    """
    #[

    @_dm.reference(category="filtering", )
    def kalman_filter(self, *args, **kwargs, ):
        r"""
        """
        return _kalmans.kalman_filter(
            self,
            *args,
            generate_period_system=_generate_period_system,
            generate_period_data=_generate_period_data,
            **kwargs,
        )

    @_dm.reference(category="filtering", )
    def neg_log_likelihood(self, *args, **kwargs, ):
        r"""
        """
        return _kalmans.neg_log_likelihood(
            self,
            *args,
            generate_period_system=_generate_period_system,
            generate_period_data=_generate_period_data,
            **kwargs,
        )

    #]


def _generate_period_system(
    t: int,
    #
    solution_v: Solution,
    y1_array: _np.ndarray,
    std_u_array: _np.ndarray,
    std_w_array: _np.ndarray,
    all_v_impact: Sequence[_np.ndarray | None] | None,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    T = solution_v.Ta
    P = solution_v.Pa
    K = solution_v.Ka
    inx_y = ~_np.isnan(y1_array[:, t], )
    Z = solution_v.Za[inx_y, :]
    H = solution_v.H[inx_y, :]
    D = solution_v.D[inx_y]
    U = solution_v.Ua
    cov_u = _np.diag(std_u_array[:, t]**2, )
    cov_w = _np.diag(std_w_array[:, t]**2, )
    v_impact = all_v_impact[t] if all_v_impact is not None else None
    return T, P, K, Z, H, D, cov_u, cov_w, v_impact, U,
    #]


def _generate_period_data(
    t,
    y_array: _np.ndarray,
    u_array: _np.ndarray,
    v_array: _np.ndarray,
    w_array: _np.ndarray,
) -> tuple[_np.ndarray, ...]:
    """
    """
    #[
    inx_y = ~_np.isnan(y_array[:, t], )
    y = y_array[inx_y, t]
    u = u_array[:, t]
    v = v_array[:, t]
    w = w_array[:, t]
    return y, u, v, w, inx_y.tolist(),
    #]

