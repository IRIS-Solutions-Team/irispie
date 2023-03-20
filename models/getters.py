
#[
from __future__ import annotations
from typing import Literal
import functools

from .. import quantities
from ..dataman import databanks
#]


class GetterMixin:
    """
    Frontend getter methods for Model objects
    """
    #[
    def _see_values_from_primary_variant(
        self,
        /,
        variant_attr: Literal["levels"] | Literal["changes"],
        kind: quantities.QuantityKind,
    ) -> dict[str, Number]:
        qid_to_name = self.create_qid_to_name()
        qids = list(quantities.generate_qids_by_kind(self._quantities, kind))
        x = self._variants[0].retrieve_values(variant_attr, qids)
        return databanks.Databank._from_dict({qid_to_name[q]:x[i, 0] for i, q in enumerate(qids)})

    def see_steady_levels(
        self,
        /,
    ) -> dict[str, Number]:
        return (
            self._see_values_from_primary_variant(variant_attr="levels", kind=quantities.QuantityKind.LOGLY_VARIABLE, )
            | self.see_parameters_stds()
        )

    def see_steady_changes(
        self,
        /,
    ) -> dict[str, Number]:
        return (
            self._see_values_from_primary_variant(variant_attr="changes", kind=quantities.QuantityKind.LOGLY_VARIABLE, )
            | self.see_parameters_stds()
        )

    see_parameters = functools.partialmethod(
        _see_values_from_primary_variant, variant_attr="levels", kind=quantities.QuantityKind.PARAMETER,
    )

    see_stds = functools.partialmethod(
        _see_values_from_primary_variant, variant_attr="levels", kind=quantities.QuantityKind.STD,
    )

    see_parameters_stds = functools.partialmethod(
        _see_values_from_primary_variant, variant_attr="levels", kind=quantities.QuantityKind.PARAMETER_OR_STD,
    )

    def get_log_status(
        self,
        /,
    ) -> dict[str, bool]:
        return {
            qty.human: qty.logly
            for qty in self._quantities if qty.kind in QuantityKind.LOGLY_VARIABLE
        }
    #]

