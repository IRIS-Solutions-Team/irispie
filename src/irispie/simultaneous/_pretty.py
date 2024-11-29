"""
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
    """
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
        """
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
    """
    """
    yield "NAME", row_names, {"align": "l"}


def _empty_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    """
    """
    yield "", ("", ) * len(row_names), {}


def _description_column(
    self,
    row_names: tuple[str, ...],
    **kwargs,
) -> Iterable[str, tuple[str], dict[str, Any]]:
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
    """
    _file_io.save_text(table.get_csv_string(), file_name, )

