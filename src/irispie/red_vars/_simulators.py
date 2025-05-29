r"""
"""


#[

from __future__ import annotations

import numpy as _np

from ..dates import Period
from ..dataslates import Dataslate
from ..fords import simulators as _simulators
from ..frames import SingleFrame

from ..progress_bars import ProgressBar

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .red_vars import RedVAR
    from typing import Iterable, Callable, Literal

#]


_rng = _np.random.default_rng()
_standard_normal = _rng.standard_normal


class Inlay:
    r"""
    """
    #[
    def simulate(
        self,
        input_db: Databox,
        span: Iterable[Period],
        #
        residuals_from_data: bool = True,
        deviation: bool = False,
        progress_bar_settings: dict = dict(title="Simulating RedVAR", ),
        **kwargs,
    ) -> Databox:
        r"""
        """
        return _simulate(
            self, input_db, span,
            draw_residuals=None,
            residuals_from_data=residuals_from_data,
            deviation=deviation,
            progress_bar_settings=progress_bar_settings,
            **kwargs,
        )

    def resample(
        self,
        input_db: Databox,
        span: Iterable[Period],
        method: Literal["monte_carlo", "bootstrap", "wild_bootstrap", ],
        progress_bar_settings: dict = dict(title="Resampling RedVAR", ),
        **kwargs,
    ) -> Databox:
        r"""
        """
        return _simulate(
            self, input_db, span,
            draw_residuals=_RESAMPLE_SIMULATOR_DISPATCH[method],
            residuals_from_data=True,
            deviation=False,
            progress_bar_settings=progress_bar_settings,
            **kwargs,
        )

    #]

def _simulate(
    self: RedVAR,
    input_db: Databox,
    span: Iterable[Period],
    #
    residuals_from_data: bool,
    draw_residuals: Callable | None,
    progress_bar_settings: dict,
    #
    target_db: Databox | None = None,
    num_variants: int | None = None,
    remove_initial: bool = False,
    prepend_input: bool = False,
    deviation: bool = False,
    show_progress: bool = False,
) -> Databox:
    r"""
    """
    span = tuple(span)
    num_variants = self.resolve_num_variants_in_context(num_variants, )
    needs_exogenous_impact = not deviation and self.has_exogenous
    needs_draw_residuals = draw_residuals is not None
    ignore_shocks = not residuals_from_data
    slatable = self.slatable_for_simulate(
        residuals_from_data=residuals_from_data,
    )
    dataslate = Dataslate.from_databox_for_slatable(
        slatable, input_db, span,
        num_variants=num_variants,
    )
    frame = SingleFrame(start=span[0], end=span[-1], )
    frame.resolve_columns(dataslate.start, )
    #
    zipped = zip(
        range(num_variants, ),
        self.iter_variants(),
        dataslate.iter_variants(),
    )
    #
    # Simulate each variant
    #=======================================================================
    progress_bar = ProgressBar(
        num_steps=num_variants,
        show_progress=show_progress,
        **progress_bar_settings,
    )
    for vid, model_v, dataslate_v in zipped:
        #
        # Resample residuals
        if needs_draw_residuals:
            draw_residuals(model_v, dataslate_v, )
        #
        # Calculate the impact of exogenous variables
        exogenous_impact = None
        if needs_exogenous_impact:
            exogenous_impact = _simulate_exogenous_impact(model_v, dataslate_v, )
        #
        # Run flat first-order simulation
        _simulators.simulate_flat(
            model_v, dataslate_v, frame,
            deviation=deviation,
            ignore_shocks=ignore_shocks,
            exogenous_impact=exogenous_impact,
        )
        progress_bar.increment()
    #=======================================================================
    #
    if remove_initial:
        dataslate.remove_initial()
    #
    # Convert all variants of the dataslate to a databox
    output_db = dataslate.to_databox()
    if prepend_input:
        output_db.prepend(input_db, span[0]-1, )
    #
    # Add to target databox
    if target_db is not None:
        output_db = target_db | output_db
    #
    return output_db


def _simulate_exogenous_impact(
    model_v: RedVAR,
    dataslate_v: Dataslate,
) -> _np.ndarray:
    r"""
    """
    #[
    B = model_v.get_system_matrices().B
    exogenous_qids = list(model_v.get_exogenous_qids())
    data_array = dataslate_v.get_data_variant()
    x_array = data_array[exogenous_qids, :]
    exogenous_impact = B @ x_array
    return exogenous_impact
    #]


def _resample_residuals_by_wild_bootstrap(
    model_v: RedVAR,
    dataslate_v: Dataslate,
    random_factor_generator: Callable[[tuple[int, int]], _np.ndarray] = _standard_normal,
) -> None:
    r"""
    """
    #[
    order = model_v.order
    residual_qids = model_v.get_residual_qids()
    data_array = dataslate_v.get_data_variant()
    residual_array = data_array[residual_qids, order:]
    num_periods = residual_array.shape[1]
    size = (1, num_periods, )
    factors = random_factor_generator(size, )
    data_array[residual_qids, order:] = factors * residual_array
    #]


def _resample_residuals_by_bootstrap(
    model_v: RedVAR,
    dataslate_v: Dataslate,
) -> None:
    r"""
    """
    #[
    ...
    #]


def _resample_residuals_by_monte_carlo(
    model_v: RedVAR,
    dataslate_v: Dataslate,
) -> None:
    r"""
    """
    #[
    ...
    #]


_RESAMPLE_SIMULATOR_DISPATCH = {
    "monte_carlo": _resample_residuals_by_monte_carlo,
    "bootstrap": _resample_residuals_by_bootstrap,
    "wild_bootstrap": _resample_residuals_by_wild_bootstrap,
}


