"""
Simultaneous model simulation inlay
"""


#[
from __future__ import annotations

import numpy as _np

from typing import Any, Literal
from .. import quantities as _quantities
from .. import has_variants as _has_variants
from .. import wrongdoings as _wrongdoings
from ..import dates as _dates
from ..databoxes.main import Databox
from ..dataslates.main import Dataslate
from ..fords import solutions as _solutions
from ..plans.simulation_plans import SimulationPlan
from ..fords import simulators as _ford_simulator
from ..period_by_period import simulators as _period_by_period_simulator
from ..stacked_time import simulators as _stacked_time_simulator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..frames import Frame
    from ..dates import Period
#]


_SIMULATOR_MODULE = {
    "first_order": _ford_simulator,
    "period_by_period": _period_by_period_simulator,
    "period": _period_by_period_simulator,
    "stacked_time": _stacked_time_simulator,
    "stacked": _stacked_time_simulator,
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
        output_parameters: bool = False,
        force_split_frames: bool = False,
        when_fails: Literal["critical", "error", "warning", "silent"] = "critical",
        #
        unpack_singleton: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> Databox | tuple[Databox, _Info]:
        r"""
        """

        num_variants = self.resolve_num_variants_in_context(num_variants, )

        base_dates = tuple(span, )

        if plan is not None and not plan.is_empty:
            plan.check_consistency(self, base_dates, )
            extra_databox_names = plan.get_databox_names()
        else:
            plan = None
            extra_databox_names = None

        slatable = self.slatable_for_simulate(
            shocks_from_data=shocks_from_data,
            stds_from_data=stds_from_data,
            parameters_from_data=parameters_from_data,
            output_parameters=output_parameters,
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

        when_fails_stream = _wrongdoings.create_stream(when_fails, "Simulation failed to complete", )
        unanticipated_shock_qids = _get_unanticipated_shock_qids(self, )
        simulator_module = _SIMULATOR_MODULE[method]

        #=======================================================================
        # Main loop over variants and over frames
        #
        out_info = []
        for vid, model_v, dataslate_v in zipped:

            input_data_array = dataslate_v.get_data_variant().copy()

            frames = simulator_module.create_frames(
                model_v, dataslate_v, plan,
                force_split_frames=force_split_frames,
            )

            info_v = {
                "method": simulator_module.METHOD_NAME,
                "frames": tuple(frames),
                "exit_status": (),
                "frame_databoxes": (),
            }

            simulator_module.simulate_initial_guess(
                model_v, dataslate_v, plan,
                **kwargs,
            )

            for frame in frames:

                simulation_header = _create_simulation_header(vid, frame, )

                # Create a full copy of the dataslate, both invariant and
                # variants. This is necessary because the frame will first prune
                # the data in place, and then remove initial and terminal data
                # and periods.
                frame_ds = dataslate_v.copy()

                # Remove unanticipated data from the frame except the first
                # simulation period for SplitFrames; keep everything unchanged
                # for SingleFrames
                frame.prune_frame_data(frame_ds, unanticipated_shock_qids, )

                exit_status = simulator_module.simulate_frame(
                    model_v, frame_ds,
                    frame=frame,
                    input_data_array=input_data_array,
                    plan=plan,
                    simulation_header=simulation_header,
                    return_info=return_info,
                    **kwargs,
                )

                if not exit_status.is_success:
                    when_fails_stream.add(f"{simulation_header}: {exit_status}", )

                # In SplitFrames, copy frame data back to the main dataslate
                # * Unanticipated shocks are written back within the frame slice
                # * Other data are writen back within the simulation slice
                # In SingleFrames, do nothing because everyrhing is kept and
                # performed within the main dataslate
                frame.write_frame_data_to_main_dataslate(
                    dataslate_v, frame_ds, unanticipated_shock_qids,
                )

                # Create a frame databox only if output info is requested
                # for performance reasons
                frame_db = (
                    frame_ds.to_databox(span="base", )
                    if return_info else None
                )

                info_v["frame_databoxes"] += (frame_db, )
                info_v["exit_status"] += (exit_status, )

            out_info.append(info_v, )

        when_fails_stream._raise()
        #
        #=======================================================================
        #
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


def _append_frame_databox(
    frame_databoxes: list[Databox],
    frame_ds: Dataslate,
) -> None:
    """
    """
    #[
    frame_db = frame_ds.to_databox(span="base", )
    frame_databoxes += (frame_db, )
    #]


def _get_unanticipated_shock_qids(self, /, ) -> tuple[int, ...]:
    """
    """
    #[
    plannable = self.get_simulation_plannable()
    name_to_qid = self.create_name_to_qid()
    unanticipated_shock_names = getattr(plannable, "can_be_endogenized_unanticipated", (), )
    return tuple(name_to_qid[i] for i in unanticipated_shock_names)
    #]


def _create_simulation_header(
    vid: str,
    frame: Frame,
) -> str:
    """
    """
    #[
    simulation_span = _dates.get_printable_span(frame.start, frame.simulation_end, )
    return f"[Variant {vid}][Periods {simulation_span}]"
    #]

