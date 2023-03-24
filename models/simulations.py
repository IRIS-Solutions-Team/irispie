"""
First-order system simulators
"""

#[
from __future__ import annotations
from IPython import embed

from typing import (Self, TypeAlias, NoReturn, Literal, )
import numpy as np_

from ..models import (core as co_, )
from ..dataman import (databanks as db_, dataslabs as ds_, )
from ..fords import (simulators as sr_, )
#]


class SimulationMixin:
    """
    """
    #[
    def simulate(
        self,
        in_databank: db_.Databank,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
        deviation: bool = False,
        prepend_input: bool = True,
    ) -> db_.Databank:
        """
        """
        ext_range, base_columns = self.get_extended_range_from_base_range(base_range)
        names = self.get_ordered_names()

        v = 0

        dataslab = ds_.Dataslab.from_databank(in_databank, names, ext_range, column=v)
        variant = self._variants[v]

        new_data = sr_.simulate_flat(
            variant.solution, self._dynamic_descriptor.solution_vectors,
            np_.copy(dataslab.data), base_columns, deviation, anticipate,
        )
        dataslab.data = new_data
        dataslab.remove_columns(base_columns[-1] - len(ext_range) + 1)
        out_databank = dataslab.to_databank()
        if prepend_input:
            under_databank = in_databank._copy()
            under_databank._clip(None, base_range[0]-1)
            out_databank._underlay(under_databank)

        out_databank = out_databank | self.get_parameters_stds()

        return out_databank
    #]

