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
    ) -> None:
        """
        """
        #
        # Choose dense or sparse array creator
        self.is_sparse = sparse_jacobian
        self._create_jacobian = (
            _create_sparse_jacobian
            if self.is_sparse else _create_dense_jacobian
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
        #
        self._map = _maps.ArrayMap(
            eids,
            eid_to_wrts,
            wrt_something,
            eid_to_rhs_offset,
            rhs_column=0,
            lhs_column_offset=0,
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
        ...

    #]


def _create_dense_jacobian(shape, diff_array, map, /, ) -> _np.ndarray:
    """
    Create Jacobian as numpy array
    """
    #[
    J = _np.zeros(shape, dtype=float, )
    J[map.lhs] = diff_array[map.rhs]
    return J
    #]


def _create_sparse_jacobian(shape, diff_array, map, /, ) -> _sp.sparse.coo_matrix:
    """
    Create Jacobian as scipy sparse matrix
    """
    #[
    J = _sp.sparse.coo_matrix(
        (diff_array[map.rhs], (map.lhs[0], map.lhs[1], )),
        shape, dtype=float,
    )
    return J
    #]

