r"""
First-order stacked-time system
"""

#[

from __future__ import annotations

import numpy as _np
import scipy as _sp

from .. import has_variants as _has_variants
from ..simultaneous.main import Simultaneous
from .. import quantities as _quantities
from ..quantities import QuantityKind, Quantity
from ..databoxes.main import Databox
from ..dataslates.main import Dataslate
from .. import dates as _dates
from ..dates import Period

from ._invariants import Invariant
from ._variants import Variant
from ..fords import covariances as _covariances
from . import _slatable_protocols as _slatable_protocols

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..simultaneous.main import Simultaneous

#]


__all__ = [
    "Stacker",
]


MarginalTupe = tuple[_np.ndarray, _np.ndarray]


class Stacker(
    _slatable_protocols.Inlay,
    _has_variants.Mixin,
    _quantities.Mixin,
):
    r"""
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    @classmethod
    def from_simultaneous(
        klass,
        model: Simultaneous,
        span: Iterable[Period],
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = Invariant.from_simultaneous(model, span, **kwargs, )
        zipped_iters = zip(
            model.iter_solution(),
            model.iter_std_name_to_value(),
            model.iter_cov_u(),
        )
        self._variants = [
            Variant.from_solution_and_stds(self._invariant, solution, std_name_to_value, cov_u, )
            for solution, std_name_to_value, cov_u in zipped_iters
        ]
        return self

    @property
    def max_lag(self, ) -> int:
        return self._invariant.max_lag

    @property
    def max_lead(self, ) -> int:
        return self._invariant.max_lead

    @property
    def base_periods(self, ) -> Iterable[Period]:
        return self._invariant.base_periods

    @property
    def stacked_vector(self, ) -> tuple[str]:
        r"""
        """
        names = self.transition_variable_names
        periods = self.base_periods
        return tuple( f"{n}[{p}]" for p in periods for n in names )

    @property
    def transition_variable_names(self, ) -> tuple[str]:
        r"""
        """
        qid_to_name = self.create_qid_to_name()
        tokens = self._invariant.stacked_solution_vectors.transition_variables
        return tuple( qid_to_name[t.qid] for t in tokens )

    def create_dataslate(
        self,
        databox: Databox,
        #
        num_variants: int | None = None,
        shocks_from_data: bool = False,
        stds_from_data: bool = False,
    ) -> Dataslate:
        r"""
        """
        num_variants = self.resolve_num_variants_in_context(num_variants, )
        slatable = self.slatable_for_marginal(
            shocks_from_data=shocks_from_data,
            stds_from_data=stds_from_data,
        )
        return Dataslate.from_databox_for_slatable(
            slatable, databox, self.base_periods,
            num_variants=num_variants,
        )


    def calculate_marginal(
        self,
        databox: Databox,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> MarginalType | list[MarginalType]:
        r"""
        """
        marginals = [
            self._calculate_marginal_for_variant(v, databox, **kwargs, )
            for v in self._variants
        ]
        return _has_variants.unpack_singleton(
            marginals, self.is_singleton,
            unpack_singleton=unpack_singleton,
        )

    def _calculate_marginal_for_variant(
        self,
        variant: Variant,
        databox: Databox,
        **kwargs,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        r"""
        """
        #
        dataslate = self.create_dataslate(databox, num_variants=1, **kwargs, )
        dataslate.logarithmize()
        data_array = dataslate.get_data_variant()
        #
        y1, u0, v0, w0 = self._extract_data_arrays(data_array, variant, )
        #
        alpha0 = variant.init_med
        cov_alpha0 = variant.init_mse
        U = variant.U
        #
        cov_u = self._construct_cov_u(data_array, )
        cov_w = self._construct_cov_w(data_array, )
        #
        TT = variant.TT
        PP = variant.PP
        RR = variant.RR
        KK = variant.KK
        #
        y1_index = _where_observations(y1, )
        y1 = y1[y1_index]
        AA = variant.AA[y1_index, :]
        BB = variant.BB[y1_index, :]
        CC = variant.CC[y1_index, :]
        DD = variant.DD[y1_index]
        HH = variant.HH[y1_index, :]
        #
        num_alpha = alpha0.size
        num_u = u0.size
        num_w = w0.size
        ABH = _np.hstack((AA, BB, HH, ))
        TP = _np.hstack((TT, PP, ))
        auw0 = _np.hstack((alpha0, u0, w0, ))
        cov_auw0 = _sp.linalg.block_diag(cov_alpha0, cov_u, cov_w, )
        #
        # Nonstochastic part of x and y
        x_nonstochastic = RR @ v0 + KK
        y_nonstochastic = CC @ v0 + DD
        #
        # Unconditional distribution
        y0 = AA @ alpha0 + BB @ u0 + HH @ w0 + y_nonstochastic
        FF = AA @ cov_alpha0 @ AA.T + BB @ cov_u @ BB.T + HH @ cov_w @ HH.T
        FF = _covariances.symmetrize(FF)
        #
        # Prediction error
        pe = y1 - y0
        #
        # Update conditional on y
        # auw_update = cov_auw0 @ ABH.T @ _np.linalg.solve(FF, pe, )
        G = _right_div(cov_auw0 @ ABH.T, FF, )
        auw_update = G @ pe
        auw1 = auw0 + auw_update
        alpha1 = auw1[:num_alpha]
        au1 = auw1[:num_alpha+num_u]
        x1 = TP @ au1 + x_nonstochastic
        #
        # cov_auw_update = cov_auw0 @ ABH.T @ _np.linalg.solve(FF, ABH @ cov_auw0, )
        cov_auw_update = G @ ABH @ cov_auw0
        cov_auw1 = cov_auw0 - cov_auw_update
        cov_auw1 = _covariances.symmetrize(cov_auw1)
        #
        if variant.has_unknown_initial:
            MM = variant.MM[y1_index, :]
            delta = self._estimate_unknown_fixed_initial(FF, MM, pe, )
            auw1_diff = self._correct_for_unknown_fixed_initial(delta, variant.Xi, AA, G, )
            auw1 = auw1 + auw1_diff
        #
        # alpha1 = auw1[:num_alpha]
        # u1 = auw1[num_alpha:num_alpha+num_u]
        # w1 = auw1[num_alpha+num_u:]
        #
        au1 = auw1[:num_alpha+num_u]
        cov_au1 = cov_auw1[:num_alpha+num_u, :num_alpha+num_u]
        #
        mean_x1 = TP @ au1 + x_nonstochastic
        cov_x1 = TP @ cov_au1 @ TP.T
        return mean_x1, cov_x1,

    def _extract_data_arrays(
        self,
        data_array: _np.ndarray,
        variant: Variant,
    ) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray, ]:
        r"""
        """
        s = self._invariant.squid
        u0 = data_array[s.u_qids, :].flatten(order="F", )
        v0 = data_array[s.v_qids, :].flatten(order="F", )
        w0 = data_array[s.w_qids, :].flatten(order="F", )
        y1 = data_array[s.y_qids, :].flatten(order="F", )
        return y1, u0, v0, w0,

    def _estimate_unknown_fixed_initial(
        self,
        FF: _np.ndarray,
        MM: _np.ndarray,
        pe: _np.ndarray,
    ) -> _np.ndarray:
        r"""
        """
        Fi_M = _left_div(FF, MM, )
        Mt_Fi_M = MM.T @ Fi_M
        Mt_Fi_pe = Fi_M.T @ pe
        delta, *_ = _np.linalg.lstsq(Mt_Fi_M, Mt_Fi_pe, rcond=None, )
        return delta

    def _correct_for_unknown_fixed_initial(
        self,
        delta: _np.ndarray,
        Xi: _np.ndarray,
        AA: _np.ndarray,
        G: _np.ndarray,
    ) -> Variant:
        r"""
        Calculate the correction to the state vector caused by the unknown fixed initial condition
        """
        # *_corr is the corrected state
        # *_diff is the difference between the corrected and uncorrected state
        #
        # alpha0_diff := alpha_corr - alpha0
        # alpha0_corr := alpha0 + Xi @ delta
        # hence alpha_diff := Xi @ delta
        alpha0_diff = Xi @ delta
        num_alpha = alpha0_diff.size
        num_auw = G.shape[0]
        num_uw = num_auw - num_alpha
        auw0_diff = _np.hstack((alpha0_diff, _np.zeros(num_uw, ), ), )
        y0_diff = AA @ alpha0_diff
        # pe = y1 - y0
        # pe_corr = y1 - y0_corr
        # pe_diff = pe_corr - pe = (y1 - y0_corr) - (y1 - y0) = y0 - y0_corr = -y0_diff
        pe_diff = -y0_diff
        # auw1 = auw0_corr + cov_auw0 @ ABH.T @ _np.linalg.solve(FF, pe_corr, )
        auw1_diff = auw0_diff + G @ pe_diff
        return auw1_diff

    def _construct_cov_u(
        self,
        data_array: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        std_u_qids = self._invariant.squid.std_u_qids
        std_u = data_array[std_u_qids, :].flatten(order="F", )
        return _np.diag(std_u**2)

    def _construct_cov_w(
        self,
        data_array: _np.ndarray,
    ) -> _np.ndarray:
        """
        """
        std_w_qids = self._invariant.squid.std_w_qids
        std_w = data_array[std_w_qids, :].flatten(order="F", )
        return _np.diag(std_w**2)

    #]


_left_div = _np.linalg.solve

def _right_div(
    A: _np.ndarray,
    B: _np.ndarray,
) -> _np.ndarray:
    r"""
    A @ inv(B)
    """
    return _left_div(B.T, A.T, ).T


def _where_observations(vector: _np.ndarray, ) -> list[int]:
    return _np.where(_np.isfinite(vector), )[0].tolist()

