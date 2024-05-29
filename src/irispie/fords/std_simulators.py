"""
"""


#[
from __future__ import annotations

from typing import (Any, Iterable, )
import wlogging as _wl

from ..dates import (Period, )
from ..dataslates.main import (Dataslate, )
#]


_LOGGER = _wl.get_colored_two_liner(__name__, level=_wl.INFO, )


_Info = dict[str, Any] | list[dict[str, Any]]


class Mixin:
    """
    """
    #[

    def multiply_stds(
        self,
        std_db: Dataslate,
        multiplier_db: Dataslate,
        span: Iterable[Period],
        /,
        *,
        num_variants: int | None = None,
        logging_level: int = _wl.INFO,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> tuple[Dataslate, _Info]:
        """
        """
        _LOGGER.set_level(logging_level, )

        num_variants \
            = self.resolve_num_variants_in_context(num_variants, )

        base_dates = tuple(span, )

        std_slatable, multiplier_slatable = self.get_slatables_for_multiply_stds()

        std_dataslate = Dataslate.from_databox_for_slatable(
            std_slatable, std_db, base_dates,
            num_variants=num_variants,
        )

        multiplier_dataslate = Dataslate.from_databox_for_slatable(
            multiplier_slatable, multiplier_db, base_dates,
            num_variants=num_variants,
        )

        zipped = zip(
            range(num_variants),
            std_dataslate.iter_variants(),
            multiplier_dataslate.iter_variants(),
        )

        #=======================================================================
        # Main loop over variants
        #
        output_info = []
        for vid, std_ds_v, multiplier_ds_v in zipped:
            output_info_v = {}
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
            output_info.append(info_v, )
            #
        #=======================================================================

        output_db = std_dataslate.to_databox()

        if target_databox is not None:
            output_db = target_databox | output_db

        is_singleton = num_variants == 1
        output_info = _has_variants.unpack_singleton(
            output_info, is_singleton,
            unpack_singleton=unpack_singleton,
        )
        return output_db, output_info

    #]

