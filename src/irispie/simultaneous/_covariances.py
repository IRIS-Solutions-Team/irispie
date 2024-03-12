"""
"""


#[
from __future__ import annotations

import numpy as _np
import scipy as _sp
import itertools as _it

from ..incidences import main as _incidences
from ..fords import covariances as _covariances
from .. import quantities as _quantities
from .. import namings as _namings
#]


class CoverianceMixin:
    """
    """

    def get_acov(
        self,
        /,
        up_to_order: int = 0,
    ) -> tuple[_np.ndarray, ..., _namings.DimensionNames]:
        """
        Asymptotic autocovariance of model variables
        """
        #
        # Combine vectors of transition and measurement tokens, and select
        # those with zero shift only using a boolex
        system_vector, boolex_zero_shift = _get_system_vector(self, )
        dimension_names = _get_dimension_names(self, system_vector, )
        #
        # Tuple element cov_by_variant[variant_index][order] is an N-by-N
        # covariance matrix
        cov_by_variant = [
            self._getv_autocov(v, boolex_zero_shift, up_to_order=up_to_order, )
            for v in self._variants
        ]
        #
        # Tuple element cov_by_order[order] is an N-by-N covariance matrix
        # (for singleton models) or an N-by-N-by-num_variants covariance
        # matrix (for nonsingleton models)
        # stack_func = _STACK_FUNC_FACTORY[self.is_singleton]
        # cov_by_order = tuple(
        #     stack_func(tuple(cov[order] for cov in cov_by_variant))
        #     for order in range(up_to_order + 1)
        # )
        cov_by_variant = self.unpack_singleton(cov_by_variant, )
        return cov_by_variant, dimension_names

    def get_acorr(
        self,
        /,
        acov: tuple[_np.ndarray, ..., tuple[str], tuple[str]] | None = None,
        **kwargs,
    ) -> tuple[_np.ndarray, ..., _namings.DimensionNames]:
        """
        Asymptotic autocorrelation of model variables
        """
        system_vector, boolex_zero_shift = _get_system_vector(self, )
        dimension_names = _get_dimension_names(self, system_vector, )
        acov_by_variant = acov
        if acov_by_variant is None:
            acov_by_variant, *_ = self.get_acov(**kwargs, )
        acov_by_variant = self.repack_singleton(acov_by_variant, )
        acorr_by_variant = [
            _covariances.acorr_from_acov(i)
            for i in acov_by_variant
        ]
        acorr_by_variant = self.unpack_singleton(acorr_by_variant, )
        return acorr_by_variant, dimension_names

    def _getv_autocov(
        self,
        variant: _variants.Variant,
        boolex_zero_shift: tuple[bool, ...] | Ellipsis,
        /,
        up_to_order: int = 0,
    ) -> _np.ndarray:
        """
        """
        def select(cov: _np.ndarray, /, ) -> _np.ndarray:
            """
            Select elements of solution vectors with zero shift only
            """
            cov = cov[boolex_zero_shift, :]
            cov = cov[:, boolex_zero_shift]
            return cov
        cov_u = self._getv_cov_u(variant, )
        cov_w = self._getv_cov_w(variant, )
        cov_by_order = _covariances.get_autocov_square(variant.solution, cov_u, cov_w, up_to_order, )
        return tuple(select(cov, ) for cov in cov_by_order)

    def get_std_unanticipated_shocks(self, /, ):
        """
        """
        std_u = [
            self._getv_std_u(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(std_u, )

    def get_std_measurement_shocks(self, /, ):
        """
        """
        std_w = [
            self._getv_std_w(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(std_w, )

    def get_cov_unanticipated_shocks(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        """
        """
        cov_u = [
            self._getv_cov_u(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(cov_u, unpack_singleton=unpack_singleton, )

    def get_cov_measurement_shocks(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        """
        """
        cov_w = [
            self._getv_cov_w(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(cov_w, unpack_singleton=unpack_singleton, )

    def _getv_std_u(self, variant, /, ):
        """
        """
        shocks = self._invariant.dynamic_descriptor.solution_vectors.unanticipated_shocks
        return _retrieve_stds(self, variant, shocks, )

    def _getv_std_w(self, variant, /, ):
        """
        """
        shocks = self._invariant.dynamic_descriptor.solution_vectors.measurement_shocks
        return _retrieve_stds(self, variant, shocks, )

    def _getv_cov_u(self, variant, /, ):
        """
        """
        stds = self._getv_std_u(variant, )
        return _np.diag(stds**2, )

    def _getv_cov_w(self, variant, /, ):
        """
        """
        stds = self._getv_std_w(variant, )
        return _np.diag(stds**2, )


def _stack_singleton(x: tuple[_np.ndarray], /, ) -> _np.ndarray:
    return x[0]


def _stack_nonsingleton(x: tuple[_np.ndarray], /, ) -> _np.ndarray:
    return _np.dstack(x)


_STACK_FUNC_FACTORY = {
    True: _stack_singleton,
    False: _stack_nonsingleton,
}


def _get_dimension_names(
    self,
    system_vector: tuple[_incidences.Token, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    """
    qid_to_name = self.create_qid_to_name()
    qid_to_logly = self.create_qid_to_logly()
    names = tuple(
        _quantities.wrap_logly(qid_to_name[tok.qid], qid_to_logly[tok.qid], )
        for tok in system_vector
    )
    return _namings.DimensionNames(rows=names, columns=names, )


def _get_system_vector(
    self,
    /,
) -> tuple[tuple[_incidences.Token, ...], tuple[bool, ...]]:
    """
    """
    system_vector = \
        self._invariant.dynamic_descriptor.solution_vectors.transition_variables \
        + self._invariant.dynamic_descriptor.solution_vectors.measurement_variables
    boolex_zero_shift = tuple(tok.shift == 0 for tok in system_vector)
    system_vector = tuple(_it.compress(system_vector, boolex_zero_shift, ))
    return system_vector, boolex_zero_shift


def _retrieve_stds(self, variant, shocks, ) -> _np.ndarray:
    """
    """
    #[
    std_qids = tuple(
        self._invariant._shock_qid_to_std_qid[t.qid]
        for t in shocks
    )
    return variant.retrieve_values("levels", std_qids, )
    #]

