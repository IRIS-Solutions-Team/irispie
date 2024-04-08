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
from .. import has_variants as _has_variants
from ..databoxes.main import (Databox, )
from ..dataslates.main import (Dataslate, )
from ..fords import simulators as _ford_simulators
from ..fords import solutions as _solutions
from ..periods import simulators as _period_simulators
from ..plans import main as _plans

from . import main as _simultaneous
#]


_SIMULATE_METHODS = {
    "first_order": _ford_simulators.simulate,
    "period": _period_simulators.simulate,
}


InfoOutput = dict[str, Any] | list[dict[str, Any]]


class Inlay:
    """
    """
    #[

    def simulate(
        self,
        input_db: Databox,
        span: Iterable[Dater],
        /,
        *,
        plan: _plans.PlanSimulate | None = None,
        method: Literal["first_order", "period", "stacked"] = "first_order",
        prepend_input: bool = True,
        target_databox: Databox | None = None,
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
        shocks_from_data: bool = True,
        logging_level: int = _wl.INFO,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> tuple[Databox, InfoOutput]:
        """
        """
        logger = _wl.get_colored_logger(__name__, level=logging_level, )
        num_variants = self.num_variants if num_variants is None else num_variants
        base_dates = tuple(span, )
        #
        extra_databox_names = None
        if plan is not None:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        #
        slatable = self.get_slatable(
            shocks_from_data=shocks_from_data,
        )
        dataslate = Dataslate.from_databox_for_slatable(
            slatable, input_db, base_dates,
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
        info = []
        simulate_method = _SIMULATE_METHODS[method]
        for vid, model_v, dataslate_v in zipped:
            #
            # Simulate and write to dataslate
            info_v = simulate_method(
                model_v, dataslate_v, vid, logger,
                plan=plan,
                **kwargs,
            )
            info.append(info_v, )

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
        info = _has_variants.unpack_singleton(
            info, self.is_singleton,
            unpack_singleton=unpack_singleton,
        )
        return output_db, info

    #]

