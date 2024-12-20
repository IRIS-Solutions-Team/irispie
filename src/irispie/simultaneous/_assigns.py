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
    r"""
    ................................................................................
    ==Class for Managing Assignments to Quantities==

    The `Inlay` class provides methods to manage assignments of custom values to
    quantities in a model. It ensures flexibility by allowing assignments through
    dictionaries or keyword arguments and validating against existing quantities.

    This class encapsulates the logic to handle assignments, process their formats,
    and enforce rules to ensure the integrity of the quantities being updated.

    Attributes:
        _variants: List of model variants managed by the instance.
        _invariant: Object containing invariant-related rules and logic.
    ................................................................................
    """

    #[

    def assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Assign Values to Quantities in a Flexible Manner==

        This method allows the assignment of custom values to quantities in a
        flexible manner, supporting input as dictionaries or keyword arguments.

        Internally, it delegates the assignment operation to `_assign`.

        ### Input arguments ###
        ???+ input "*args"
            Arbitrary positional arguments, typically dictionaries containing
            quantity-value pairs.
        ???+ input "**kwargs"
            Arbitrary keyword arguments specifying quantity-value pairs.

        ### Returns ###
        ???+ returns "None"
            This method does not return any value.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            obj.assign({"quantity1": 10, "quantity2": 20}, quantity3=30)
        ```
        ................................................................................
        """
        assigned_keys, nonexistent_keys = self._assign(*args, **kwargs, )

    def assign_strict(
        self,
        *args,
        **kwargs,
    ) -> None:
        r"""
        ................................................................................
        ==Strict Assignment Method with Validation==

        This method assigns custom values to quantities but enforces strict
        validation. If any nonexistent quantities are specified, it raises an
        exception.

        ### Input arguments ###
        ???+ input "*args"
            Arbitrary positional arguments, typically dictionaries containing
            quantity-value pairs.
        ???+ input "**kwargs"
            Arbitrary keyword arguments specifying quantity-value pairs.

        ### Returns ###
        ???+ returns "None"
            This method does not return any value.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            try:
                obj.assign_strict({"quantity1": 10, "nonexistent_quantity": 99})
            except IrisPieCritical as e:
                print(e)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Internal Assignment Logic==

        This internal method handles the core logic for assigning values to quantities
        from dictionaries or keyword arguments. It validates the input and manages
        quantity mappings, ensuring consistent processing across variants.

        ### Input arguments ###
        ???+ input "*args"
            Arbitrary positional arguments, typically dictionaries containing
            quantity-value pairs.
        ???+ input "**kwargs"
            Arbitrary keyword arguments specifying quantity-value pairs.

        ### Returns ###
        ???+ returns "tuple"
            A tuple containing:
            - Assigned keys.
            - Nonexistent keys.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            result = obj._assign({"quantity1": 10, "quantity2": 20}, quantity3=30)
            print(result)
        ```
        ................................................................................
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
        r"""
        ................................................................................
        ==Update Steady Auto-Values==

        Updates auto-values for steady state variants using invariant rules.
        This method ensures that all variants adhere to the latest steady-state
        constraints.

        ### Input arguments ###
        ???+ input "None"
            This method does not take any input arguments.

        ### Returns ###
        ???+ returns "None"
            This method does not return any value.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            obj.update_steady_autovalues()
        ```
        ................................................................................
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
    r"""
    ................................................................................
    ==Rekey a Dictionary Using a Key Mapping==

    Transforms the keys in a dictionary using a provided key mapping. Nonexistent
    keys are collected and returned for further handling.

    ### Input arguments ###
    ???+ input "dict_to_rekey"
        The dictionary whose keys need to be transformed.
    ???+ input "old_key_to_new_key"
        A mapping of old keys to their corresponding new keys.

    ### Returns ###
    ???+ returns "tuple"
        A tuple containing:
        - The transformed dictionary.
        - A tuple of keys that were successfully transformed.
        - A tuple of keys that were not found in the mapping.

    ### Example for a Function ###
    ```python
        transformed_dict, assigned_keys, nonexistent_keys = _rekey_dict(
            {"old_key": "value"}, {"old_key": "new_key"}
        )
        print(transformed_dict)
    ```
    ................................................................................
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
    r"""
    ................................................................................
    ==Prepare Custom Value Iterator==

    Converts a value or list of values into an iterator for processing variants.
    Supports handling of ellipsis for unchanged values.

    ### Input arguments ###
    ???+ input "value"
        A single value or a list of values to be processed.

    ### Returns ###
    ???+ returns "Iterator"
        An iterator yielding tuples of values, with ellipsis included for
        unchanged fields.

    ### Example for a Function ###
    ```python
        iter = _prepare_custom_value_iter([1, 2, 3])
        print(list(iter))
    ```
    ................................................................................
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
