"""
# First-order unsolved system matrices

## Unsolved system

$$
A E[x_t] + B E[x_{t-1}] + C + D u_t + E v_t = 0 \\
$$

$$
F y_t + G x_t + H + J w_t = 0
$$
"""


#[
from __future__ import annotations

import dataclasses as _dc
import numpy as _np

from . import descriptors as _descriptors
from ..simultaneous import _flags as _flags
#]


@_dc.dataclass
class System:
    """
    Unsolved system matrices
    """
    #
    # Transition equations
    #
    A: _np.ndarray | None = None
    B: _np.ndarray | None = None
    C: _np.ndarray | None = None
    D: _np.ndarray | None = None
    E: _np.ndarray | None = None
    #
    # Measurement equations
    #
    F: _np.ndarray | None = None
    G: _np.ndarray | None = None
    H: _np.ndarray | None = None
    J: _np.ndarray | None = None

    def __init__(
        self,
        descriptor: _descriptors.Descriptor,
        data_array: _np.ndarray,
        steady_array: _np.ndarray,
        model_flags: _flags.ModelFlags,
        data_array_lagged: _np.ndarray | None,
        column_offset: int,
        /,
    ) -> None:
        """
        """
        #
        # Differentiate and evaluate constant
        #
        td, tc = descriptor.aldi_context.eval_to_arrays(
            data_array, column_offset, steady_array,
        )

        smap = descriptor.system_map
        svec = descriptor.system_vectors

        self.A = _np.zeros(svec.shape_A_excl_dynid, dtype=float)
        self.A[smap.A.lhs] = td[smap.A.rhs]
        self.A = _np.vstack((self.A, smap.dynid_A))

        self.B = _np.zeros(svec.shape_B_excl_dynid, dtype=float)
        self.B[smap.B.lhs] = td[smap.B.rhs]
        self.B = _np.vstack((self.B, smap.dynid_B))

        if model_flags.is_linear:
            self.C = _np.zeros(svec.shape_C_excl_dynid, dtype=float)
            self.C[smap.C.lhs] = tc[smap.C.rhs, 0]
            self.C = _np.concatenate((self.C, smap.dynid_C))
        else:
            tokens = descriptor.system_vectors.transition_variables
            logly = descriptor.system_vectors.transition_variables_are_logly
            xi = _get_vector(descriptor, data_array, tokens, logly, column_offset, )
            xi_lagged = _get_vector(descriptor, data_array_lagged, tokens, logly, column_offset, )
            self.C = -(self.A @ xi + self.B @ xi_lagged)

        self.D = _np.zeros(svec.shape_D_excl_dynid, dtype=float)
        self.D[smap.D.lhs] = td[smap.D.rhs]
        self.D = _np.vstack((self.D, smap.dynid_D))

        self.E = _np.zeros(svec.shape_E_excl_dynid, dtype=float)
        self.E[smap.E.lhs] = td[smap.E.rhs]
        self.E = _np.vstack((self.E, smap.dynid_E))

        self.F = _np.zeros(svec.shape_F, dtype=float)
        self.F[smap.F.lhs] = td[smap.F.rhs]

        self.G = _np.zeros(svec.shape_G, dtype=float)
        self.G[smap.G.lhs] = td[smap.G.rhs]

        if model_flags.is_linear:
            self.H = _np.zeros(svec.shape_H, dtype=float)
            self.H[smap.H.lhs] = tc[smap.H.rhs, 0]
        else:
            tokens = descriptor.system_vectors.transition_variables
            logly = descriptor.system_vectors.transition_variables_are_logly
            xi = _get_vector(descriptor, data_array, tokens, logly, column_offset, )
            tokens = descriptor.system_vectors.measurement_variables
            logly = descriptor.system_vectors.measurement_variables_are_logly
            y = _get_vector(descriptor, data_array, tokens, logly, column_offset, )
            self.H = -(self.F @ y + self.G @ xi)

        self.J = _np.zeros(svec.shape_J, dtype=float)
        self.J[smap.J.lhs] = td[smap.J.rhs]


def _get_vector(
    descriptor: _descriptors.Descriptor,
    data_array: _np.ndarray,
    tokens: Iterable[_incidence.Token],
    logly: Iterable[bool],
    column_offset: int,
    /,
) -> _np.ndarray:
    """
    """
    #[
    logly = list(logly)
    rows = tuple(tok.qid for tok in tokens)
    columns = tuple(column_offset + tok.shift for tok in tokens)
    x = data_array[rows, columns]
    xx = _np.copy(x)
    x[logly] = _np.log(x[logly])
    return x
    #]

