"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np
import wlogging as _wl

from .. import quantities as _quantities
from ..databoxes import main as _databoxes
from ..fords import simulators as _ford_simulators
from ..fords import solutions as _solutions
from ..periods import simulators as _period_simulators
from ..plans import main as _plans
from ..dataslates import main as _dataslates

from . import main as _simultaneous
#]


_dataslate_constructor = _dataslates.Dataslate.from_databox_for_slatable


class SimulateMixin:
    """
    """
    #[

    def simulate(
        self,
        input_db: _databoxes.Databox,
        span: Iterable[Dater],
        /,
        *,
        plan: _plans.PlanSimulate | None = None,
        method: Literal["first_order", "period", "stacked"] = "first_order",
        prepend_input: bool = True,
        target_databox: _databoxes.Databox | None = None,
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
        shocks_from_databox: bool = True,
        logging_level: int = _wl.INFO,
        **kwargs,
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
        """
        logger = _wl.get_colored_logger(__name__, level=logging_level, )
        num_variants = self.num_variants if num_variants is None else num_variants
        base_dates = tuple(span, )
        #
        work_db = input_db.shallow()
        if not shocks_from_databox:
            shock_names = self.get_names(kind=_quantities.ANY_SHOCK, )
            work_db.remove(shock_names, strict=False, )
        #
        extra_databox_names = None
        if plan is not None:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        #
        dataslate = _dataslate_constructor(
            self, work_db, base_dates,
            num_variants=num_variants,
            extra_databox_names=extra_databox_names,
        )
        #
        zipped = zip(
            range(num_variants, ),
            self.iter_variants(),
            dataslate.iter_variants(),
        )
        #
        #=======================================================================
        # Main loop over variants
        for vid, model_v, dataslate_v in zipped:
            #
            # Simulate and write to dataslate
            _SIMULATE_METHODS[method](
                model_v, dataslate_v, vid,
                plan=plan,
                **kwargs,
            )
        #=======================================================================
        #
        # Remove initial and terminal condition data (all lags and leads
        # before and after the simulation span)
        if remove_terminal:
            dataslate.remove_terminal()
        if remove_initial:
            dataslate.remove_initial()
        #
        # Convert all variants of the dataslate to a databox
        output_db = dataslate.to_databox()
        if prepend_input:
            output_db.prepend(input_db, base_dates[0]-1, )
        #
        # Add to custom databox
        if target_databox is not None:
            output_db = target_databox | output_db
        #
        # Simulation info
        info = {}
        #
        return output_db, info

    #]


def _simulate_first_order(
    model: _simultaneous.Simultaneous,
    dataslate: _dataslates.Dataslate,
    vid: int,
    **kwargs,
) -> None:
    """
    """
    _ford_simulators.simulate_flat(
        model._variants[0].solution,
        model._solution_vectors,
        dataslate,
        vid,
        **kwargs,
    )


_SIMULATE_METHODS = {
    "first_order": _simulate_first_order,
    "period": _period_simulators.simulate,
}


