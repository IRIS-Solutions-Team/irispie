"""
"""


#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )
import dataclasses as dc_
import numpy as np_
import scipy as sp_

from ..aldi import (differentiators as ad_, maps as am_, )
from .. import (equations as eq_, quantities as qu_, incidence as in_, )
#]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Frontend
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


class Descriptor:
    """
    """
    #[
    __slots__ = (
        "_num_rows", "_num_columns", "_map", "_qid_to_logly", "_aldi_context",
        "_create_jacobian", "is_sparse",
    )

    def __init__(self, /, **kwargs, ) -> None:
        """
        """
        self.is_sparse = kwargs.get("sparse_jacobian", False)
        if self.is_sparse:
            self._create_jacobian = self._create_sparse_jacobian
        else:
            self._create_jacobian = self._create_dense_jacobian

    @classmethod
    def for_flat(
        cls, 
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        qid_to_logly: dict[int, bool],
        function_context: dict[str, Callable] | None,
        /,
        **kwargs,
    ) -> NoReturn:
        """
        """
        self = cls(**kwargs, )
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
        diff_array = self._aldi_context.eval_diff_to_array(data_context, self._qid_to_logly, L, )
        return self._create_jacobian(diff_array, self._map, )

    def _create_dense_jacobian(self, diff_array, map, /, ) -> np_.ndarray:
        """
        Create Jacobian as numpy array
        """
        J = np_.zeros(
            (self._num_rows, self._num_columns, ),
            dtype=float,
        )
        J[map.lhs] = diff_array[map.rhs]
        return J

    def _create_sparse_jacobian(self, diff_array, map, /, ) -> sp_.sparse.coo_matrix:
        """
        Create Jacobian as scipy sparse matrix
        """
        J = sp_.sparse.coo_matrix(
            (diff_array[map.rhs], (map.lhs[0], map.lhs[1], )),
            (self._num_rows, self._num_columns, ),
            dtype=float,
        )
        return J


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


