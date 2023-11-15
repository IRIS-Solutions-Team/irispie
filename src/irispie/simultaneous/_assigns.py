"""
Assign custom values to quantities
"""


#[
from __future__ import annotations

from typing import (Any, )
from collections.abc import (Iterable, Iterator, )
from numbers import (Number, )

from ..conveniences import iterators as _iterators
from ..databoxes import main as _databoxes
#]


class AssignMixin:
    """
    """
    #[

    def assign(
        self,
        /,
        *args,
        **kwargs,
    ) -> None:
        """
        Assign parameters from dicts or from keyword arguments
        """
        for arg in args:
            self.assign(**arg, )
        if not kwargs:
            return
        name_to_qid = self.create_name_to_qid()
        qid_to_name = self.create_qid_to_name()
        qid_to_custom_values = _rekey_dict(kwargs, name_to_qid, )
        qid_to_custom_values_iter = _databoxes.Databox.iter_variants(
            qid_to_custom_values,
            item_iterator=_prepare_custom_value_iter,
        )
        for variant, values in zip(self._variants, qid_to_custom_values_iter, ):
            variant.update_values_from_dict(values, )
            self._enforce_assignment_rules(variant, )

    #]


def _rekey_dict(
    dict_to_rekey: dict,
    old_key_to_new_key: dict,
    /,
    garbage_key=None,
) -> dict[int, Any]:
    """
    """
    #[
    new_dict = {
        old_key_to_new_key.get(key, garbage_key): value
        for key, value in dict_to_rekey.items()
    }
    if garbage_key in new_dict:
        del new_dict[garbage_key]
    return new_dict
    #]


_UNCHANGED_VALUE = (..., ..., )


def _prepare_custom_value_iter(
    value: Any,
    /,
) -> Iterator[tuple[Number | EllipsisType, Number | EllipsisType]]:
    """
    Resolve custom values: value = [a, b, c] means variants, a = (1, 2) means level and change, ... means keep unchanged
    """
    #[
    is_iterable = (
        isinstance(value, Iterable, )
        and not isinstance(value, str, )
        and not isinstance(value, bytes, )
        and not isinstance(value, tuple, )
    )
    value = value if is_iterable else [value, ]
    return _iterators.exhaust_then_last(value, _UNCHANGED_VALUE, )
    #]

