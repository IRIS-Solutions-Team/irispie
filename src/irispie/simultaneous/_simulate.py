"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np

from ..databoxes import main as _databoxes
from ..fords import simulators as _ford_simulators
from ..fords import solutions as _solutions
from ..periods import simulators as _period_simulators
from ..plans import main as _plans
from .. import dataslates as _dataslates

from . import main as _simultaneous
#]


_DATASLATE_VARIANT_ITERATOR = \
    _dataslates.HorizontalDataslate.iter_variants_from_databox_for_slatable


class SimulateMixin:
    """
    """
    #[

    def simulate(
        self,
        in_databox: _databoxes.Databox,
        span: Iterable[Dater],
        /,
        *,
        plan: _plans.PlanSimulate | None = None,
        method: Literal["first_order", "period", "stacked"] = "first_order",
        prepend_input: bool = True,
        target_databox: _databoxes.Databox | None = None,
        num_variants: int | None = None,
        **kwargs,
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
        """
        num_variants = (
            num_variants
            if num_variants is not None
            else self.num_variants
        )
        base_dates = tuple(span, )
        if plan is not None:
            plan.check_consistency(self, base_dates, )
        #
        out_dataslates = []
        zipped = zip(
            range(num_variants, ),
            self.iter_variants(),
            _DATASLATE_VARIANT_ITERATOR(self, in_databox, base_dates, )
        )
        #
        #=======================================================================
        # Main loop over variants
        for vid, mdi, dsi in zipped:
            #
            # Simulate and write to dataslate
            _SIMULATE_METHODS[method](
                mdi, dsi, vid,
                plan=plan,
                **kwargs,
            )
            dsi.remove_terminal()
            out_dataslates.append(dsi, )
        #=======================================================================
        #
        # Combine resulting dataslates into a databox
        out_db = _dataslates.multiple_to_databox(out_dataslates, )
        if prepend_input:
            out_db.prepend(in_databox, base_dates[0]-1, )
        out_db = out_db | self.get_parameters_stds()
        #
        # Add to custom databox
        if target_databox is not None:
            out_db = target_databox | out_db
        #
        # Simulation info
        info = {}
        #
        return out_db, info

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
        model.get_solution_vectors(),
        dataslate,
        vid,
        **kwargs,
    )


_SIMULATE_METHODS = {
    "first_order": _simulate_first_order,
    "period": _period_simulators.simulate,
}


