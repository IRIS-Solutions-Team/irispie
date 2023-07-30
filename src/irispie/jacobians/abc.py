"""
"""

#[
import numpy as _np
import scipy as _sp
from typing import (Protocol, Callable, )
from collections.abc import (Iterable, )

from ..aldi import (differentiators as _ad, maps as _am, )
from .. import (equations as _eq, quantities as _qu, incidence as _in, )
#]


class Jacobian:
    """
    # Describe the Jacobian matrix of a system of equations:
    * _shape -- shape of the Jacobian matrix (num_rows, num_columns)
    * _map -- mapping from (row, column) to (equation, quantity)
    * _aldi_context -- context for algorithmic differentiator wrt levels
    * _change_aldi_context -- context for algorithmic differentiator wrt changes
    * _array_creator -- function to create a dense or sparse Jacobian matrix
    * is_sparse -- whether the Jacobian matrix is sparse
    """
    #[
    atom_factory: _ad.AtomFactoryProtocol | None = None

    def __init__(
        self, 
        equations: _eq.Equations,
        wrt_qids: Iterable[int],
        qid_to_logly: dict[int, bool],
        /,
        *,
        custom_functions: dict[str, Callable] | None = None,
        **kwargs,
    ) -> None:
        """
        """
        #
        # Choose dense or sparse array creator
        self.is_sparse = kwargs.get("sparse_jacobian", False)
        self._create_jacobian = (
            _create_sparse_jacobian
            if self.is_sparse else _create_dense_jacobian
        )
        #
        # Extract eids from equations
        eids = tuple(eqn.id for eqn in equations)
        #
        # Collect the qids w.r.t. which each equation is to be differentiated
        wrt_qids = tuple(wrt_qids)
        eid_to_wrt_qids = {
            eqn.id: list(_generate_flat_wrt_qids_in_equation(eqn, wrt_qids, ))
            for eqn in equations
        }
        #
        # Create the map from eids to rhs offsets; the offset is the number
        # of rows in the Jacobian matrix that precede the equation
        eid_to_rhs_offset = _am.create_eid_to_rhs_offset(eids, eid_to_wrt_qids, )
        #
        self._shape = len(eids), len(wrt_qids),
        #
        self._map = _am.ArrayMap.for_equations(
            eids,
            eid_to_wrt_qids,
            wrt_qids,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self._aldi_context = _ad.Context(
            equations,
            self.atom_factory,
            eid_to_wrts=eid_to_wrt_qids,
            qid_to_logly=qid_to_logly,
            num_columns_to_eval=1,
            custom_functions=custom_functions,
        )


def _create_dense_jacobian(shape, diff_array, map, /, ) -> _np.ndarray:
    """
    Create Jacobian as numpy array
    """
    J = _np.zeros(shape, dtype=float, )
    J[map.lhs] = diff_array[map.rhs]
    return J


def _create_sparse_jacobian(shape, diff_array, map, /, ) -> _sp.sparse.coo_matrix:
    """
    Create Jacobian as scipy sparse matrix
    """
    J = _sp.sparse.coo_matrix(
        (diff_array[map.rhs], (map.lhs[0], map.lhs[1], )),
        shape, dtype=float,
    )
    return J


def _generate_flat_wrt_qids_in_equation(equation, all_wrt_qids):
    """
    Generate subset of the wrt_qids that occur in this equation (no matter what shift)
    """
    return ( 
        qid for qid in all_wrt_qids 
        if _in.is_qid_in_tokens(equation.incidence, qid)
    )


