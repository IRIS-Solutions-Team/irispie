"""
Jacobians for steady systems
"""


#[
from __future__ import annotations

from typing import (Protocol, )
import numpy as _np
import scipy as _sp

from ..incidences import main as _incidence

from . import _base
#]



class SteadyJacobianProtocol(Protocol, ):
    """
    """
    #[
    def eval(
        self,
        steady_array: _np.ndarray,
        /,
    ) -> _np.ndarray:
        ...
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
        token: _incidence.Token,
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
        token: _incidence.Token,
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
        token: _incidence.Token,
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


class FlatSteadyJacobian(_base.Jacobian, ):
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


class NonflatSteadyJacobian(_base.Jacobian, ):
    """
    """
    #[
    NONFLAT_STEADY_SHIFT: int = 1
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
        k = self.NONFLAT_STEADY_SHIFT
        return _np.vstack((
            _np.hstack((A, B, )),
            _np.hstack((A, B+k*A, )),
        ))
    #]


