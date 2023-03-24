
#[
from __future__ import annotations
from IPython import embed

import functools as ft_
import json as js_
from typing import Literal

from .. import quantities as qu_
from ..dataman import databanks as db_
#]

def _decorate_output_format(func):
    """
    """
    #[
    def _wrapper(*args, **kwargs):
        output, output_format = func(*args, **kwargs)
        return db_.DATABANK_OUTPUT_FORMAT_RESOLUTION[output_format](output)
    return _wrapper
    #]


class GetterMixin:
    """
    Frontend getter methods for Model objects
    """
    #[
    def _get_values_from_primary_variant(
        self,
        /,
        variant_attr: Literal["levels"] | Literal["changes"],
        kind: qu_.QuantityKind,
    ) -> dict[str, Number]:
        qid_to_name = self.create_qid_to_name()
        qids = list(qu_.generate_qids_by_kind(self._quantities, kind))
        x = self._variants[0].retrieve_values(variant_attr, qids)
        xxx= db_.Databank._from_dict({ qid_to_name[q]: float(x[i, 0]) for i, q in enumerate(qids) })
        return xxx

    @_decorate_output_format
    def get_steady_levels(
        self,
        /,
    ) -> dict[str, Number]:
        return (
            self._get_values_from_primary_variant(variant_attr="levels", kind=qu_.QuantityKind.LOGLY_VARIABLE, )
            | self.get_parameters_stds()
        )

    @_decorate_output_format
    def get_steady_changes(
        self,
        /,
    ) -> dict[str, Number]:
        return (
            self._get_values_from_primary_variant(variant_attr="changes", kind=qu_.QuantityKind.LOGLY_VARIABLE, )
            | self.get_parameters_stds()
        )

    @_decorate_output_format
    def get_parameters(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=qu_.QuantityKind.PARAMETER), kwargs.get("output", "Databank")

    @_decorate_output_format
    def get_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=qu_.QuantityKind.STD), kwargs.get("output", "Databank")

    @_decorate_output_format
    def get_parameters_stds(
        self,
        /,
        **kwargs,
    ) -> dict[str, Number]:
        return self._get_values_from_primary_variant(variant_attr="levels", kind=qu_.QuantityKind.PARAMETER_OR_STD), kwargs.get("output", "Databank")

    @_decorate_output_format
    def get_log_status(
        self,
        /,
    ) -> dict[str, bool]:
        return {
            qty.human: qty.logly
            for qty in self._quantities if qty.kind in QuantityKind.LOGLY_VARIABLE
        }
    #]

