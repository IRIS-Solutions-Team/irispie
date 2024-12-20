"""
Provides utilities for creating and manipulating formatted tables, particularly 
for representing steady-state data and related attributes.
"""


#[
from __future__ import annotations

from prettytable import (PrettyTable, )
import functools as _ft

from .. import quantities as _quantities
from .. import file_io as _file_io

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, )
    from collections.abc import (Iterable, )
#]


_DEFAULT_ROUND_TO = 16


class Inlay:
    r"""
    ................................................................................
    ==Class: Inlay==

    Facilitates the generation of formatted tables for steady-state values and related 
    data. The `Inlay` class provides functionality to dynamically construct tables 
    with various columns, including names, descriptions, kinds, log statuses, and 
    steady-state levels and changes.

    Attributes:
        - `_invariant.quantities`: Quantities managed by the instance.
    ................................................................................
    """
    #[

    def create_steady_table(
        self,
        columns: Iterable[str] = ("name", "steady_level", "steady_change", ),
        kind: int = _quantities.ANY_VARIABLE | _quantities.PARAMETER,
        names: tuple[str, ...] | None = None,
        save_to_csv_file: str | None = None,
        **kwargs,
    ) -> PrettyTable:
        r"""
        ................................................................................
        ==Method: create_steady_table==

        Generates a formatted table for steady-state data using the `PrettyTable` 
        library. The table's columns can be customized based on user preferences.

        ### Input arguments ###
        ???+ input "columns: Iterable[str] = ('name', 'steady_level', 'steady_change',)"
            A list of column names to include in the table. Supported columns are 
            dynamically determined.
        ???+ input "kind: int = _quantities.ANY_VARIABLE | _quantities.PARAMETER"
            The kind of quantities to include in the table.
        ???+ input "names: tuple[str, ...] | None = None"
            A tuple of specific quantity names to include. If `None`, all relevant 
            quantities are included based on the kind.
        ???+ input "save_to_csv_file: str | None = None"
            The file name to save the table as a CSV file. If `None`, no file is saved.
        ???+ input "**kwargs"
            Additional keyword arguments for customizing table generation.

        ### Returns ###
        ???+ returns "PrettyTable"
            A `PrettyTable` instance containing the formatted table.

        ### Example ###
        ```python
            table = obj.create_steady_table(columns=["name", "description"])
            print(table)
        ```
        ................................................................................
        """
        column_constructors = [
            _COLUMN_CONSTRUCTORS[column.lower().strip()]
            for column in columns
        ]
        row_names = tuple(names) if names is not None else self.get_names(kind=kind, )
        table = PrettyTable()
        for constructor in column_constructors:
            for header, values, settings in constructor(self, row_names, **kwargs, ):
                table.add_column(header, values, **settings, )
        if save_to_csv_file:
            _save_pretty_table_to_csv_file(table, save_to_csv_file, )
        return table

    #]


def _name_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _name_column==

    Constructs the "NAME" column for the steady-state table. Includes the names of 
    the quantities.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _name_column(obj, row_names)
    ```
    ................................................................................
    """
    yield "NAME", row_names, {"align": "l"}


def _empty_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _empty_column==

    Constructs an empty column for the steady-state table. This function can be used 
    to create a spacer or placeholder column with no content.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header (empty string), row values (empty strings), and 
        settings for PrettyTable.

    ### Example ###
    ```python
        column = _empty_column(obj, row_names)
    ```
    ................................................................................
    """
    yield "", ("", ) * len(row_names), {}


def _description_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _description_column==

    Constructs the "DESCRIPTION" column for the steady-state table. Includes the 
    descriptions of the quantities.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _description_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    name_to_description = _quantities.create_name_to_description(self._invariant.quantities, )
    descriptions = tuple(
        name_to_description.get(name, "")
        for name in row_names
    )
    yield "DESCRIPTION", descriptions, {"align": "l"}
    #]


def _kind_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _kind_column==

    Constructs the "KIND" column for the steady-state table. Includes the kinds of 
    the quantities (e.g., parameter, variable).

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _kind_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    name_to_kind = _quantities.create_name_to_kind(self._invariant.quantities, )
    kinds = tuple(
        str(name_to_kind.get(name, "")).split(".", )[-1].upper()
        for name in row_names
    )
    yield "KIND", kinds, {"align": "l"}
    #]


def _log_status_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _log_status_column==

    Constructs the "LOG_STATUS" column for the steady-state table. Indicates whether 
    quantities are stored in logarithmic scale.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _log_status_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    name_to_logly = _quantities.create_name_to_logly(self._invariant.quantities, )
    logly = tuple(
        name_to_logly.get(name, None, )
        for name in row_names
    )
    yield "LOG_STATUS", logly, {"align": "r"}
    #]


def _comparison_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _comparison_column==

    Constructs the "COMPARISON" column for the steady-state table. Indicates the type 
    of comparison (ratio or difference) based on the logarithmic status of quantities.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _comparison_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    name_to_logly = _quantities.create_name_to_logly(self._invariant.quantities, )
    comparisons = tuple(
        "ratio/" if name_to_logly.get(name, False) else "diff-"
        for name in row_names
    )
    yield "COMPARISON", comparisons, {"align": "c"}
    #]


def _steady_level_column(
    self,
    row_names: tuple[str, ...],
    round_to: int = _DEFAULT_ROUND_TO,
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _steady_level_column==

    Constructs the "STEADY_LEVEL" column for the steady-state table. Displays the 
    steady-state levels of quantities, rounded to the specified precision.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "round_to: int = _DEFAULT_ROUND_TO"
        The number of decimal places to round the steady-state levels.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _steady_level_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    def _display_value(value: Real, ):
        try:
            return round(value, round_to, )
        except:
            return value
    iter_levels = (self.get_steady_levels() | self.get_parameters()).iter_variants()
    num_variants = self.num_variants
    header = "STEADY_LEVEL_{}" if num_variants > 1 else "STEADY_LEVEL"
    for i, levels in zip(range(num_variants), iter_levels):
        yield (
            header.format(i, ),
            tuple( _display_value(levels[name]) for name in row_names ),
            {"align": "r"},
        )
    #]


def _compare_steady_value(
    orig_value_iterator,
    self,
    row_names,
    round_to: int = _DEFAULT_ROUND_TO,
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _compare_steady_value==

    Constructs comparison columns for steady-state values. Computes differences or 
    ratios between variants based on the logarithmic status of quantities.

    ### Input arguments ###
    ???+ input "orig_value_iterator"
        The original iterator function for steady-state levels or changes.
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "round_to: int = _DEFAULT_ROUND_TO"
        The number of decimal places to round the comparison values.
    ???+ input "**kwargs"
        Additional arguments for the original iterator function.

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _compare_steady_value(_steady_level_column, obj, row_names)
    ```
    ................................................................................
    """
    #[
    name_to_logly = _quantities.create_name_to_logly(self._invariant.quantities, )
    _, base_values, *_ = next(orig_value_iterator(self, row_names, **kwargs, ), )
    def comparison_func_plain(x, y):
        return x-y if x is not None and y is not None else None
    def comparison_func_log(x, y):
        return x/y if x is not None and y is not None else None
    comparison_func = {
        name: comparison_func_plain if not name_to_logly.get(name, False)
        else comparison_func_log
        for name in row_names
    }
    #
    def _compare(values: Iterable[Real], ) -> Iterable[Real]:
        return tuple(
            comparison_func[name](value, base_value, ) if value != "" else ""
            for name, value, base_value in zip(row_names, values, base_values, )
        )
    #
    # Skip comparison of first variant to first variant
    next(orig_value_iterator(self, row_names, **kwargs, ))
    #
    for header, values, settings in orig_value_iterator(self, row_names, **kwargs, ):
        yield "COMPARE_" + header, _compare(values, ), settings
    #]


def _steady_change_column(
    self,
    row_names: tuple[str, ...],
    round_to: int = _DEFAULT_ROUND_TO,
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    r"""
    ................................................................................
    ==Function: _steady_change_column==

    Constructs the "STEADY_CHANGE" column for the steady-state table. Displays the 
    steady-state changes of quantities, rounded to the specified precision.

    ### Input arguments ###
    ???+ input "self"
        The instance invoking the function.
    ???+ input "row_names: tuple[str, ...]"
        The names of the rows to include in the column.
    ???+ input "round_to: int = _DEFAULT_ROUND_TO"
        The number of decimal places to round the steady-state changes.
    ???+ input "**kwargs"
        Additional arguments (unused).

    ### Returns ###
    ???+ returns "Iterable[str, tuple[str], dict[str, Any]]"
        Yields the column header, row values, and settings for PrettyTable.

    ### Example ###
    ```python
        column = _steady_change_column(obj, row_names)
    ```
    ................................................................................
    """
    #[
    def _display_value(value: Real, ):
        try:
            return round(value, round_to, )
        except:
            return value
    iter_changes = (self.get_steady_changes()).iter_variants()
    num_variants = self.num_variants
    header = "STEADY_CHANGE_{}" if num_variants > 1 else "STEADY_CHANGE"
    for i, changes in zip(range(num_variants), iter_changes):
        yield (
            header.format(i, ),
            tuple( _display_value(changes.get(name, ""), ) for name in row_names ),
            {"align": "r"},
        )
    #]


_COLUMN_CONSTRUCTORS = {
    "name": _name_column,
    "empty": _empty_column,
    "description": _description_column,
    "kind": _kind_column,
    "log_status": _log_status_column,
    "logly": _log_status_column,
    "log": _log_status_column,
    "comparison": _comparison_column,
    "steady_level": _steady_level_column,
    "steady_change": _steady_change_column,
    "compare_steady_level": _ft.partial(_compare_steady_value, _steady_level_column, ),
    "compare_steady_change": _ft.partial(_compare_steady_value, _steady_change_column, ),
}


def _save_pretty_table_to_csv_file(
    table: PrettyTable,
    file_name: str,
) -> None:
    r"""
    ................................................................................
    ==Function: _save_pretty_table_to_csv_file==

    Saves the contents of a `PrettyTable` instance to a CSV file. This function is 
    useful for exporting formatted tables to external files.

    ### Input arguments ###
    ???+ input "table: PrettyTable"
        The `PrettyTable` instance containing the table data.
    ???+ input "file_name: str"
        The name of the CSV file to save the table.

    ### Returns ###
    (No return value)

    ### Example ###
    ```python
        _save_pretty_table_to_csv_file(table, "output.csv")
    ```
    ................................................................................
    """
    _file_io.save_text(table.get_csv_string(), file_name, )

