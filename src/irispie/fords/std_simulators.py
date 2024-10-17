"""
"""


#[
from __future__ import annotations

from typing import TYPE_CHECKING

from .. import has_variants as _has_variants
from ..dataslates.main import Dataslate
from ..databoxes.main import Databox

if TYPE_CHECKING:
    from typing import Any, Iterable
    from ..dates import Period
    _Info = dict[str, Any] | list[dict[str, Any]]
#]


class Mixin:
    """
    """
    #[

    def vary_stds(
        self,
        multiplier_db: Databox | None,
        std_db: Databox | None,
        span: Iterable[Period],
        num_variants: int | None = None,
        target_db: Databox | None = None,
        #
        unpack_singleton: bool = True,
        return_info: bool = False,
        **kwargs,
    ) -> Databox | tuple[Databox, _Info]:
        """
        """
        num_variants \
            = self.resolve_num_variants_in_context(num_variants, )

        base_dates = tuple(span, )

        std_slatable, multiplier_slatable = self.get_slatables_for_multiply_stds()

        multiplier_ds = Dataslate.from_databox_for_slatable(
            multiplier_slatable, multiplier_db or Databox(), base_dates,
            num_variants=num_variants,
        )

        std_ds = Dataslate.from_databox_for_slatable(
            std_slatable, std_db or Databox(), base_dates,
            num_variants=num_variants,
        )

        zipped = zip(
            range(num_variants),
            std_ds.iter_variants(),
            multiplier_ds.iter_variants(),
        )

        #=======================================================================
        # Main loop over variants
        #
        out_info = []
        for vid, std_ds_v, multiplier_ds_v in zipped:
            out_info_v = {}
            #
            std_array = std_ds_v.get_data_variant()
            multiplier_array = multiplier_ds_v.get_data_variant()
            names = std_ds_v._invariant.names
            #
            zipped = zip(names, std_array, multiplier_array)
            #
            for names, std_row, multiplier_row in zipped:
                std_row *= multiplier_row
            #
            out_info.append(out_info_v, )
            #
        #=======================================================================

        out_db = std_ds.to_databox()

        if target_db is not None:
            out_db = target_db | out_db

        if return_info:
            out_info = _has_variants.unpack_singleton(
                out_info,
                num_variants == 1,
                unpack_singleton=unpack_singleton,
            )
            return out_db, out_info
        else:
            return out_db

    #]

