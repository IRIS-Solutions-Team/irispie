"""
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp

from .. import equations as _equations
from .. import quantities as _quantities
from ..incidences import main as _incidences
from ..aldi.differentiators import (AtomFactoryProtocol, Context, )
from ..aldi import maps as _maps
from ..aldi.maps import (ArrayMap, )

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Protocol, Callable, )
    from collections.abc import (Iterable, )

#]


class _Jacobian:
    """
    """
    #[

    _atom_factory: AtomFactoryProtocol = ...

    def _populate_map(self, *args, **kwargs, ) -> None:
        raise NotImplementedError

    def eval(self, *args, **kwargs, ) -> _np.ndarray:
        raise NotImplementedError

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        wrt_something: Iterable[Any],
        qid_to_logly: dict[int, bool],
        /,
        *,
        context: dict[str, Callable] | None = None,
        initial: _np.ndarray | None = None,
        columns_to_eval: Iterable[int] | None = None,
        num_columns_to_eval: int = 1,
        **kwargs,
    ) -> None:
        """
        """
        self._columns_to_eval = _np.array(columns_to_eval, dtype=int, ) if columns_to_eval is not None else None
        self._num_columns_to_eval = num_columns_to_eval if columns_to_eval is None else len(columns_to_eval)
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
        eid_to_rhs_offset = _maps.create_eid_to_rhs_offset(eids, eid_to_wrts, )
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
        #
        self.sparse_pattern = None
        self._matrix = None

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

    def _create_jacobian_matrix(self, diff_array, /, ) -> Any:
        """
        Create Jacobian as scipy sparse matrix
        """
        jacobian_matrix = self._initialize_jacobian_matrix()
        jacobian_matrix[self._map.lhs] = diff_array[self._map.rhs]
        return jacobian_matrix

    #]


class DenseJacobian(_Jacobian):
    """
    """
    #[

    def _initialize_jacobian_matrix(self, /, ) -> _np.ndarray:
        """
        Initialize the Jacobian matrix as a dense numpy array
        """
        return _np.zeros(self._shape, dtype=float, )

    #]


class SparseJacobian(_Jacobian):
    """
    """
    #[

    def __init__(self, *args, terminator: Terminator | None, **kwargs, ) -> None:
        """
        """
        super().__init__(*args, **kwargs, )
        self.sparse_pattern = self._map.lhs

    def _initialize_jacobian_matrix(self, /, ) -> _sp.sparse.csc_matrix:
        """
        Initialize the Jacobian matrix as a sparse scipy array
        """
        num_entries = len(self.sparse_pattern[0])
        return _sp.sparse.csc_matrix(
            ([0]*num_entries, self.sparse_pattern, ),
            shape=self._shape, dtype=float,
        )
    #]

