"""
"""


#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )
import dataclasses as dc_
import numpy as np_

from ..aldi import (differentiators as ad_, maps as am_, )
from .. import (equations as eq_, quantities as qu_, incidence as in_, )
#]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Frontend
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


@dc_.dataclass
class Descriptor:
    """
    """
    #[
    _map: am_.ArrayMap | None = None
    _qid_to_logly: dict[int, bool] | None = None,
    _aldi_context: ad_.Context | None = None
    _num_rows: int | None = None,
    _num_columns: int | None = None,

    @classmethod
    def for_flat(
        cls, 
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        qid_to_logly: dict[int, bool],
        function_context: dict[str, Callable] | None,
        /,
    ) -> NoReturn:
        """
        """
        self = cls()
        #
        eids = [ eqn.id for eqn in equations ]
        all_wrt_qids = [ qty.id for qty in quantities ]
        eid_to_wrt_qids = {
            eqn.id: list(_generate_flat_wrt_qids_in_equation(eqn, all_wrt_qids, ))
            for eqn in equations
        }
        eid_to_rhs_offset = am_.create_eid_to_rhs_offset(eids, eid_to_wrt_qids, )
        #
        self._num_rows = len(eids)
        self._num_columns = len(all_wrt_qids)
        self._qid_to_logly = qid_to_logly
        #
        self._map = am_.ArrayMap.for_equations(
            eids,
            eid_to_wrt_qids,
            all_wrt_qids,
            eid_to_rhs_offset,
        )
        #
        self._aldi_context = ad_.Context.for_equations(
            ad_.FlatSteadyAtom,
            equations,
            eid_to_wrt_qids,
            1,
            function_context,
        )
        #
        return self

    def eval(
        self,
        data_context: np_.ndarray,
        L: np_.ndarray,
        /,
    ) -> np_.ndarray:
        """
        """
        J = self._initialize_jacobian()
        diff_array = self._aldi_context.eval_diff_to_array(data_context, self._qid_to_logly, L, )
        J[self._map.lhs] = diff_array[self._map.rhs]
        return J

    def _initialize_jacobian(self, /, ) -> np_.ndarray:
        return np_.zeros((self._num_rows, self._num_columns, ), dtype=float, )


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Backend
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def _generate_flat_wrt_qids_in_equation(equation, all_wrt_qids):
    """
    Generate subset of the wrt_qids that occur in this equation (no matter what shift)
    """
    return ( 
        qid for qid in all_wrt_qids 
        if in_.is_qid_in_tokens(equation.incidence, qid)
    )


