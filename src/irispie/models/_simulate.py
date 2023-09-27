"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, Any, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np

from ..databanks import main as _databanks
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
        in_databank: _databanks.Databank,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
        deviation: bool = False,
        plan: _plans.Plan | None = None,
        prepend_input: bool = True,
        add_to_databank: _databanks.Databank | None = None,
    ) -> tuple[_databanks.Databank, dict[str, Any]]:
        """
        """
        dataslates = tuple(
            _dataslates.Dataslate(self, in_databank, base_range, plan=plan, slate=i, )
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
            dataslate.remove_columns(dataslate.base_columns[-1] - dataslate.num_periods + 1)
        #
        out_db = _dataslates.multiple_to_databank(dataslates)
        if prepend_input:
            out_db.prepend(in_databank, base_range[0]-1, )
        out_db = out_db | self.get_parameters_stds()
        #
        # Add to custom databank
        #
        if add_to_databank is not None:
            out_db = add_to_databank | out_db
        #
        info = {"dataslates": dataslates, }
        return out_db, info

    #]


