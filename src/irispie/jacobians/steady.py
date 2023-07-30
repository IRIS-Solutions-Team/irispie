"""
Jacobians for steady systems
"""


#[
import numpy as _np
import scipy as _sp

from ..aldi import (differentiators as _ad, maps as _am, )
from .. import (equations as _eq, quantities as _qu, incidence as _in, )
from . import (abc as _abc, )
from .. import (incidence as _in, )
from ..evaluators import (steady as _es, )
#]


class _AtomFactory:
    """
    """
    #[
    def create_diff_for_token(*args, **kwargs):
        """
        """
        ...

    @staticmethod
    def create_data_index_for_token(
        token: _in.Token,
        columns_to_eval: tuple[int, int],
        /,
    ) -> tuple[int, slice]:
        """
        """
        return (
            token.qid,
            slice(columns_to_eval[0], columns_to_eval[1]+1),
        )
    #]


class _FlatAtomFactory(_AtomFactory, ):
    """
    """
    #[
    @staticmethod
    def create_diff_for_token(
        token: _in.Token,
        wrt_qids: tuple[int],
        /,
    ) -> _np.ndarray:
        """
        """
        try:
            index = wrt_qids.index(token.qid)
            diff = _np.zeros((len(wrt_qids), 1))
            diff[index] = 1
            return diff
        except:
            return 0
    #]


class _NonflatAtomFactory(_AtomFactory, ):
    """
    """
    #[
    @staticmethod
    def create_diff_for_token(
        token: _in.Token,
        wrt_qids: dict[int, tuple[int]],
        /,
    ) -> _np.ndarray:
        """
        """
        try:
            index = wrt_qids.index(token.qid)
            diff = _np.zeros((len(wrt_qids), 2))
            diff[index, :] = 1, token.shift
            return diff
        except:
            return 0
    #]


class FlatSteadyJacobian(_abc.Jacobian, ):
    """
    """
    #[
    atom_factory = _FlatAtomFactory

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, steady_array, )
        return self._create_jacobian(self._shape, diff_array, self._map, )
    #]


class NonflatSteadyJacobian(_abc.Jacobian, ):
    """
    """
    #[
    atom_factory = _NonflatAtomFactory

    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        """
        """
        diff_array = self._aldi_context.eval_diff_to_array(steady_array, steady_array, )
        A = self._create_jacobian(self._shape, diff_array[:, 0:1], self._map, )
        B = self._create_jacobian(self._shape, diff_array[:, 1:2], self._map, )
        k = _es.NONFLAT_SHIFT
        return _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))
    #]


