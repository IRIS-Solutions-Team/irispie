"""
"""


#[
from __future__ import annotations

from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )
import dataclasses as dc_
import numpy as np_

from ..aldi import (maps as ma_, )
#]


@dc_.dataclass
class Descriptor:
    """
    """
    #[
    jacobian_map: ma_.ArrayMap | None = None
    aldi_context: ad_.Context | None = None

    @classmethod
    def for_flat(self, equations, wrt_qids, /,) -> NoReturn:

        eids = ...
        eid_to_rhs_offset = ...
        eid_to_wrt_tokens = ...
        all_wrt_tokens = ...

        self.map = ma_.vstack_array_maps(
            ma_.ArrayMap.for_equation(
                eid_to_wrt_tokens[eid],
                all_wrt_tokens,
                eid_to_rhs_offset[eid],
                lhs_row,
            )
            for lhs_row, eid in enumerate(eids)
        )

        self.aldi_context = ad_.Context.for_equations(
                ...
        )


