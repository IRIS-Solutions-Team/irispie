"""
Assign custom values to quantities
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, Iterable )

from ..wrongdoings import (IrisPieCritical, )
from ..conveniences import iterators as _iterators
from ..databoxes.main import (Databox, )

if TYPE_CHECKING:
    from typing import (Any, )
    from collections.abc import (Iterator, )
    from numbers import (Real, )
#]


# TODO: Use in Sequentials


class Inlay:
    """
    """
    #[

    def assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Assign custom values to quantities
        """
        assigned_keys, nonexistent_keys = self._assign(*args, **kwargs, )

    def assign_strict(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Assign custom values to quantities
        """
        assigned_keys, nonexistent_keys = self._assign(*args, **kwargs, )
        if nonexistent_keys:
            message = ("Cannot assign these names (nonexistent in the model object): ", ) + nonexistent_keys
            raise IrisPieCritical(message, )

    def _assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Assign parameters from dicts or from keyword arguments
        """
        dict_to_assign = {}
        dict_to_assign.update(*args, )
        dict_to_assign.update(kwargs, )
        if not dict_to_assign:
            return (), (),
        #
        name_to_qid = self.create_name_to_qid()
        qid_to_name = self.create_qid_to_name()
        qid_to_custom_values, assigned_keys, nonexistent_keys \
            = _rekey_dict(dict_to_assign, name_to_qid, )
        qid_to_custom_values_iter = Databox.iter_variants(
            qid_to_custom_values,
            item_iterator=_prepare_custom_value_iter,
        )
        for variant, values in zip(self._variants, qid_to_custom_values_iter, ):
            variant.update_values_from_dict(values, )
            self._enforce_assignment_rules(variant, )
        #
        return assigned_keys, nonexistent_keys

    def update_steady_autovalues(self, ) -> None:
        """
        """
        if not self._invariant.update_steady_autovalues_in_variant:
            return
        for variant in self._variants:
            self._invariant.update_steady_autovalues_in_variant(variant, )
    #]


def _rekey_dict(
    dict_to_rekey: dict,
    old_key_to_new_key: dict,
) -> tuple[dict[int, Any], tuple[str, ...], tuple[str, ...]]:
    """
    """
    #[
    new_dict = {}
    assigned_keys = set()
    nonexistent_keys = set()
    for key, value in dict_to_rekey.items():
        try:
            new_key = old_key_to_new_key[key]
            new_dict[new_key] = value
            assigned_keys.add(key)
        except KeyError:
            nonexistent_keys.add(key)
    return new_dict, tuple(assigned_keys), tuple(nonexistent_keys)
    #]


_UNCHANGED_VALUE = (..., ..., )


def _prepare_custom_value_iter(
    value: Any,
    /,
) -> Iterator[tuple[Real | EllipsisType, Real | EllipsisType]]:
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

