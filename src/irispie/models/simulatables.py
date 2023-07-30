"""
First-order system simulators
"""


#[
from __future__ import annotations
# from IPython import embed

from typing import (Self, TypeAlias, NoReturn, Literal, Protocol, runtime_checkable)
from collections.abc import (Iterable, )
import numpy as _np

from ..dataman import (databanks as _db, dataslabs as _ds, )
from ..fords import (simulators as _sr, )
from . import (variants as _va, )
#]


class _SimulatableProtocol(Protocol, ):
    """
    """
    #[
    num_variants: int
    _variants: Iterable[_va.Variant]

    def get_extended_range_from_base_range(): ...
    def get_ordered_names(): ...
    def get_solution_vectors(): ...
    #]


class Simulatable:
    """
    """
    #[
    def simulate(
        self: _SimulatableProtocol,
        in_databank: _db.Databank,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
        deviation: bool = False,
        prepend_input: bool = True,
    ) -> _db.Databank:
        """
        """
        ext_range, base_columns = self.get_extended_range_from_base_range(base_range)
        names = self.get_ordered_names()

        dataslabs = tuple(
            _ds.Dataslab.from_databank(in_databank, names, ext_range, column=i) 
            for i in range(self.num_variants)
        )

        for variant, dataslab in zip(self._variants, dataslabs):
            new_data = _sr.simulate_flat(
                variant.solution, self.get_solution_vectors(),
                _np.copy(dataslab.data), base_columns, deviation, anticipate,
            )
            dataslab.data = new_data
            dataslab.remove_columns(base_columns[-1] - len(ext_range) + 1)

        out_databank = _ds.multiple_to_databank(dataslabs)
        if prepend_input:
            under_databank = in_databank._copy()
            under_databank._clip(None, base_range[0]-1)
            out_databank._underlay(under_databank)

        out_databank = out_databank | self.get_parameters_stds()

        return out_databank
    #]


