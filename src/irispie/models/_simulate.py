"""
First-order system simulators
"""


#[
from __future__ import annotations

from typing import (Self, TypeAlias, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np

from ..databanks import main as _databanks
from ..fords import simulators as _simulators
from .. import dataslabs as _dataslabs
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
        prepend_input: bool = True,
    ) -> _databanks.Databank:
        """
        """

        dataslabs = tuple(
            _dataslabs.Dataslab.from_databank_for_simulation(self, in_databank, base_range, column=i, )
            for i in range(self.num_variants)
        )

        qid_to_logly = self.create_qid_to_logly()
        boolex_logly = tuple(
            qid_to_logly[qid] or False
            for qid in range(dataslabs[0].num_rows)
        )

        for variant, dataslab in zip(self._variants, dataslabs):
            new_data = dataslab.copy_data()
            new_data = _simulators.simulate_flat(
                variant.solution, self.get_solution_vectors(), boolex_logly,
                new_data, dataslab.base_columns, deviation, anticipate,
            )
            dataslab.data = new_data
            dataslab.remove_columns(dataslab.base_columns[-1] - dataslab.num_ext_periods + 1)

        out_databank = _dataslabs.multiple_to_databank(dataslabs)
        if prepend_input:
            out_databank.prepend(in_databank, base_range[0]-1, )

        out_databank = out_databank | self.get_parameters_stds()

        return out_databank
    #]


