"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np

from ..databoxes import main as _databoxes
from ..fords import simulators as _simulators
from ..plans import main as _plans
from .. import dataslates as _dataslates
#]


class SimulateMixin:
    """
    """
    #[

    def simulate(
        self,
        in_databox: _databoxes.Databox,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
        deviation: bool = False,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        target_databox: _databoxes.Databox | None = None,
    ) -> tuple[_databoxes.Databox, dict[str, Any]]:
        """
        """
        input_slatables = (self, )
        if plan is not None:
            plan.check_consistency(self, range, )
            input_slatables += (plan, )
        #
        # Create dataslates from input data, one for each variant
        #
        dataslates = tuple(
            _dataslates.HorizontalDataslate.for_slatables(
                input_slatables, in_databox, base_range, variant=i,
            )
            for i in range(self.num_variants)
        )
        #
        qid_to_logly = self.create_qid_to_logly()
        boolex_logly = tuple(
            qid_to_logly[qid] or False
            for qid in range(dataslates[0].num_rows)
        )
        #
        for variant, dataslate in zip(self._variants, dataslates):
            new_data = dataslate.copy_data()
            new_data = _simulators.simulate_flat(
                variant.solution, self.get_solution_vectors(), boolex_logly,
                new_data, dataslate.base_columns, deviation, anticipate,
            )
            dataslate.data = new_data
            dataslate.remove_periods_from_end(dataslate.base_columns[-1] - dataslate.num_periods + 1)
        #
        # Combine resulting dataslates into a databox
        #
        out_db = _dataslates.multiple_to_databox(dataslates, )
        if prepend_input:
            out_db.prepend(in_databox, base_range[0]-1, )
        out_db = out_db | self.get_parameters_stds()
        #
        # Add to custom databox
        #
        if target_databox is not None:
            out_db = target_databox | out_db
        #
        info = {}
        return out_db, info

    #]


