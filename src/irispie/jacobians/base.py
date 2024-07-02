"""
"""


#[
from __future__ import annotations

from typing import (Any, Protocol, Callable, )
from collections.abc import (Iterable, )
import numpy as _np
import scipy as _sp

from .. import equations as _equations
from .. import quantities as _quantities
from ..incidences import main as _incidences
from ..aldi import differentiators as _differentiators
from ..aldi import maps as _maps
#]


class Jacobian:
    """
    """
    #[

    _create_map = ...
    _calculate_shape = ...

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        wrt_something: Iterable[Any],
        qid_to_logly: dict[int, bool],
        /,
        *,
        context: dict[str, Callable] | None = None,
        first_column_to_eval: int | None = None,
        num_columns_to_eval: int = 1,
        sparse_jacobian: bool = False,
        initial: _np.ndarray | None = None,
    ) -> None:
        """
        """
        #
        # Choose dense or sparse array creator
        self._jacobian_matrix = initial
        self.is_sparse = sparse_jacobian
        self.num_columns_to_eval = num_columns_to_eval
        self._create_jacobian_matrix = (
            self._create_sparse_jacobian_matrix
            if self.is_sparse else self._create_dense_jacobian_matrix
        )
        #
        # Extract eids from equations
        eids = tuple(eqn.id for eqn in equations)
        wrt_something = tuple(wrt_something)
        #
        # Collect w.r.t. items (tokens, qids) which each equation is to be
        # differentiated
        eid_to_wrts = self._create_eid_to_wrts(equations, wrt_something, )
        #
        # Collect all tokens in the euqations, and find the minimum shift
        all_tokens = set(_equations.generate_all_tokens_from_equations(equations, ), )
        min_shift = _incidences.get_min_shift(all_tokens, )
        self._first_column_to_eval = -min_shift
        #
        # Create the map from eids to rhs offsets; the offset is the number
        # of rows in the Jacobian matrix that precede the equation
        eid_to_rhs_offset = \
            _maps.create_eid_to_rhs_offset(eids, eid_to_wrts, )
        #
        self._shape = len(eids), len(wrt_something),
        self._shape = self._calculate_shape(eids, wrt_something, num_columns_to_eval, )
        #
        self._map = self._create_map(
            eids,
            eid_to_wrts,
            wrt_something,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
            num_columns_to_eval=num_columns_to_eval,
        )
        #
        # self is the Atom factory
        atom_factory = self
        self._aldi_context = _differentiators.Context(
            atom_factory,
            equations,
            eid_to_wrts=eid_to_wrts,
            qid_to_logly=qid_to_logly,
            context=context,
        )

    def _create_eid_to_wrts(
        self,
        equations: Iterable[_equations.Equation],
        wrts: Iterable[Any],
        /,
    ) -> dict[int, tuple[Any, ...]]:
        """
        """
        raise NotImplementedError

    def _create_dense_jacobian_matrix(self, diff_array, /, ) -> _np.ndarray:
        """
        Create Jacobian as numpy array
        """
        if self._jacobian_matrix is None:
            self._jacobian_matrix = _np.zeros(self._shape, dtype=float, )
        self._jacobian_matrix[self._map.lhs] = diff_array[self._map.rhs]
        return self._jacobian_matrix.copy()

    def _create_sparse_jacobian_matrix(self, diff_array, /, ) -> _sp.sparse.coo_matrix:
        """
        Create Jacobian as scipy sparse matrix
        """
        return _sp.sparse.coo_matrix(
            (diff_array[self._map.rhs], (self._map.lhs[0], self._map.lhs[1], )),
            self._shape, dtype=float,
        )

