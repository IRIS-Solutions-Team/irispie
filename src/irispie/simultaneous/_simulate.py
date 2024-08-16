"""
Simultaneous model simulation inlay
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np
import wlogging as _wl

from .. import quantities as _quantities
from .. import has_variants as _has_variants
from ..dates import (Period, )
from ..databoxes.main import (Databox, )
from ..dataslates.main import (Dataslate, )
from ..fords import solutions as _solutions
from ..fords import simulators as _ford_simulator
from ..period_by_period import simulators as _period_by_period_simulator
from ..stacked_time import simulators as _stacked_time_simulator
from ..plans.simulation_plans import (SimulationPlan, )
#]


_LOGGER = _wl.get_colored_logger(__name__, level=_wl.INFO, )


_SIMULATION_METHOD_DISPATCH = {
    "first_order": _ford_simulator.simulate,
    "period_by_period": _period_by_period_simulator.simulate,
    "period": _period_by_period_simulator.simulate,
    "stacked_time": _stacked_time_simulator.simulate,
    "stacked": _stacked_time_simulator.simulate,
}


_Info = dict[str, Any] | list[dict[str, Any]]


class Inlay:
    """
    """
    #[

    def simulate(
        self,
        in_db: Databox,
        span: Iterable[Period],
        /,
        *,
        plan: SimulationPlan | None = None,
        method: Literal["first_order", "period_by_period", "period", "stacked_time", "stacked", ] = "first_order",
        prepend_input: bool = True,
        target_db: Databox | None = None,
        num_variants: int | None = None,
        remove_initial: bool = True,
        remove_terminal: bool = True,
        shocks_from_data: bool = True,
        stds_from_data: bool = True,
        parameters_from_data: bool = False,
        logging_level: int = _wl.INFO,
        #
        unpack_singleton: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> tuple[Databox, _Info]:
        """
        """
        _LOGGER.set_level(logging_level, )

        num_variants \
            = self.resolve_num_variants_in_context(num_variants, )
        _LOGGER.debug(f"Running {num_variants} variants")

        base_dates = tuple(span, )

        if plan is not None and not plan.is_empty:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        else:
            plan = None
            extra_databox_names = None

        slatable = self.get_slatable_for_simulate(
            shocks_from_data=shocks_from_data,
            stds_from_data=stds_from_data,
            parameters_from_data=parameters_from_data,
        )

        dataslate = Dataslate.from_databox_for_slatable(
            slatable, in_db, base_dates,
            num_variants=num_variants,
            extra_databox_names=extra_databox_names,
        )

        zipped = zip(
            range(num_variants, ),
            self.iter_variants(),
            dataslate.iter_variants(),
        )

        simulation_func = _SIMULATION_METHOD_DISPATCH[method]

        #=======================================================================
        # Main loop over variants
        #
        out_info = []
        for vid, model_v, dataslate_v in zipped:
            info_v = simulation_func(
                model_v, dataslate_v, plan, vid,
                logger=_LOGGER,
                return_info=return_info,
                **kwargs,
            )
            out_info.append(info_v, )
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
        out_db = dataslate.to_databox()
        if prepend_input:
            out_db.prepend(in_db, base_dates[0]-1, )
        #
        # Add to custom databox
        if target_db is not None:
            out_db = target_db | out_db
        #
        if return_info:
            is_singleton = num_variants == 1
            out_info = _has_variants.unpack_singleton(
                out_info, is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return out_db, out_info
        else:
            return out_db
    #]

