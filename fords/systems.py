"""
# First-order system and solution

## Unsolved system

$$
A E[x_t] + B E[x_{t-1}] + C + D v_t = 0 \\
F y_t + G x_t + H + J w_t = 0
$$
"""


#[
from __future__ import annotations
# from IPython import embed

import dataclasses as dc_
import numpy as np_ 

from . import descriptors as de_
#]


@dc_.dataclass
class System:
    """
    Unsolved system matrices
    """
    #
    # Transition equations
    #
    A: np_.ndarray | None = None
    B: np_.ndarray | None = None
    C: np_.ndarray | None = None
    D: np_.ndarray | None = None
    #
    # Measurement equations
    # F y + G xi + H + J e = 0
    #
    F: np_.ndarray | None = None
    G: np_.ndarray | None = None
    H: np_.ndarray | None = None
    J: np_.ndarray | None = None

    @classmethod
    def for_descriptor(
        cls,
        descriptor: de_.Descriptor,
        logly_context: dict[int, bool],
        value_context: np_.ndarray,
        /,
    ) -> NoReturn:
        """
        """
        # Differentiate and evaluate constant
        td, tc = descriptor.aldi_context.eval_to_arrays(value_context, logly_context)

        smap = descriptor.system_map
        svec = descriptor.system_vectors

        self = cls()

        self.A = np_.zeros(svec.shape_AB_excl_dynid, dtype=float)
        self.A[smap.A.lhs] = td[smap.A.rhs]
        self.A = np_.vstack((self.A, smap.dynid_A))

        self.B = np_.zeros(svec.shape_AB_excl_dynid, dtype=float)
        self.B[smap.B.lhs] = td[smap.B.rhs]
        self.B = np_.vstack((self.B, smap.dynid_B))

        self.C = np_.zeros(svec.shape_C_excl_dynid, dtype=float)
        self.C[smap.C.lhs] = tc[smap.C.rhs]
        self.C = np_.vstack((self.C, smap.dynid_C))

        self.D = np_.zeros(svec.shape_D_excl_dynid, dtype=float)
        self.D[smap.D.lhs] = td[smap.D.rhs]
        self.D = np_.vstack((self.D, smap.dynid_D))

        self.F = np_.zeros(svec.shape_F, dtype=float)
        self.F[smap.F.lhs] = td[smap.F.rhs]

        self.G = np_.zeros(svec.shape_G, dtype=float)
        self.G[smap.G.lhs] = td[smap.G.rhs]

        self.H = np_.zeros(svec.shape_H, dtype=float)
        self.H[smap.H.lhs] = tc[smap.H.rhs]

        self.J = np_.zeros(svec.shape_J, dtype=float)
        self.J[smap.J.lhs] = td[smap.J.rhs]

        return self


