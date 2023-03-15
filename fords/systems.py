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

from IPython import embed
import dataclasses
import numpy 

from .descriptors import Descriptor
#]


@dataclasses.dataclass
class System:
    """
    Unsolved system matrices
    """
    # Transition equations
    A: numpy.ndarray | None = None
    B: numpy.ndarray | None = None
    C: numpy.ndarray | None = None
    D: numpy.ndarray | None = None
    # Measurement equations
    E: numpy.ndarray | None = None
    B: numpy.ndarray | None = None
    C: numpy.ndarray | None = None
    D: numpy.ndarray | None = None

    @classmethod
    def for_descriptor(
        cls,
        descriptor: Descriptor,
        logly_context: dict[int, bool],
        value_context: numpy.ndarray,
        /,
    ) -> NoReturn:
        """
        """
        # Differentiate and evaluate constant
        tt = descriptor.system_differn_context.eval(value_context, logly_context)
        td = numpy.vstack([x.diff for x in tt])
        tc = numpy.vstack([x.value for x in tt])

        smap = descriptor.system_map
        svec = descriptor.system_vectors

        self = cls()

        self.A = numpy.zeros(svec.shape_AB_excl_dynid, dtype=float)
        self.A[smap.A.lhs] = td[smap.A.rhs]
        self.A = numpy.vstack((self.A, smap.dynid_A))

        self.B = numpy.zeros(svec.shape_AB_excl_dynid, dtype=float)
        self.B[smap.B.lhs] = td[smap.B.rhs]
        self.B = numpy.vstack((self.B, smap.dynid_B))

        self.C = numpy.zeros(svec.shape_C_excl_dynid, dtype=float)
        self.C[smap.C.lhs] = tc[smap.C.rhs]
        self.C = numpy.vstack((self.C, smap.dynid_C))

        self.D = numpy.zeros(svec.shape_D_excl_dynid, dtype=float)
        self.D[smap.D.lhs] = td[smap.D.rhs]
        self.D = numpy.vstack((self.D, smap.dynid_D))

        self.F = numpy.zeros(svec.shape_F, dtype=float)
        self.F[smap.F.lhs] = td[smap.F.rhs]

        self.G = numpy.zeros(svec.shape_G, dtype=float)
        self.G[smap.G.lhs] = td[smap.G.rhs]

        self.H = numpy.zeros(svec.shape_H, dtype=float)
        self.H[smap.H.lhs] = tc[smap.H.rhs]

        self.J = numpy.zeros(svec.shape_J, dtype=float)
        self.J[smap.J.lhs] = td[smap.J.rhs]

        return self


