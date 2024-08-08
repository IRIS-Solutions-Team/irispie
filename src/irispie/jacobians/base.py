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
from ..aldi.differentiators import (AtomFactoryProtocol, Context, )
from ..aldi import maps as _maps
#]


class Jacobian:
    """
    """
    #[

    _atom_factory: AtomFactoryProtocol = ...

    _populate_map: Callable = ...

    eval: Callable = ...

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        wrt_something: Iterable[Any],
        qid_to_logly: dict[int, bool],
        /,
        *,
        context: dict[str, Callable] | None = None,
        sparse_jacobian: bool = False,
        initial: _np.ndarray | None = None,
        num_columns_to_eval: int = 1,
    ) -> None:
        """
        """
        self._num_columns_to_eval = num_columns_to_eval
        self._jacobian_matrix = initial
        self.is_sparse = sparse_jacobian
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
        # Create the map from eids to rhs offsets; the offset is the number
        # of rows in the Jacobian matrix that precede the equation
        eid_to_rhs_offset = \
            _maps.create_eid_to_rhs_offset(eids, eid_to_wrts, )
        #
        self._populate_shape(eids, wrt_something, )
        self._populate_map(
            eids,
            eid_to_wrts,
            wrt_something,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
        )
        #
        self._aldi_context = Context(
            self._atom_factory,
            equations,
            eid_to_wrts=eid_to_wrts,
            qid_to_logly=qid_to_logly,
            context=context,
        )

    def _populate_shape(
        self,
        eids: Collection[int],
        wrt_something: Collection[Any],
    ) -> None:
        """
        """
        self._shape = len(eids)*self._num_columns_to_eval, len(wrt_something),

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

