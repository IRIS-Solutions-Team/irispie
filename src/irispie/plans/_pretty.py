"""
"""


#[
from __future__ import annotations

from typing import (Self, Any, )
import numpy as _np
import prettytable as _pt
import itertools as _it

from .. import pages as _pages
from ..databoxes.main import (Databox, )
from ..series.main import (Series, )
from ..dates import (Period, )
#]



_TABLE_FIELDS_NO_VALUE = ("NAME", "PERIOD(S)", "REGISTER", "TRANSFORM", )
_TABLE_FIELDS_WITH_VALUE = _TABLE_FIELDS_NO_VALUE + ("VALUE", )


class PrettyMixin:
    @property
    def pretty(self, /, ) -> _pt.PrettyTable:
        """
        """
        return self.get_pretty()

    @property
    def pretty_full(self, /, ) -> str:
        """
        """
        return self.get_pretty(full=True, )

    def get_pretty(
        self,
        /,
        db: Databox | None = None,
    ) -> _pt.PrettyTable:
        """
        """
        table = _pt.PrettyTable()
        if db is None:
            table.field_names = _TABLE_FIELDS_NO_VALUE
        else:
            table.field_names = _TABLE_FIELDS_WITH_VALUE
        table.align = "r"
        table.align["NAME"] = "l"
        for r in self._registers:
            if getattr(self, f"_{r}_register"):
                _add_register_to_table(
                    table,
                    self.base_span,
                    getattr(self, f"_{r}_register"),
                    r,
                    db,
                )
        return table

    @_pages.reference(
        category="information",
    )
    def print_table(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Print the `SimulationPlan` as a table==

    self.print_table()


### Input arguments ###

???+ input "self"
    `SimulationPlan` to be printed on the screen, with one row showing
    exogenized or endogenized data points grouped by the name and dates.


### Returns ###

Returns no value; the table is printed on the screen.

................................................................................
        """
        print(self.get_pretty(*args, **kwargs, ), )

    def pretty_print(self, *args, **kwargs, ) -> None:
        """
        """
        print(self.get_pretty(*args, **kwargs, ), )

    def get_pretty_string(self, *args, **kwargs, ) -> str:
        """
        """
        return self.get_pretty(*args, **kwargs, ).get_string()


def _add_register_to_table(
    table,
    base_span,
    register,
    action,
    db: Databox | None = None,
) -> None:
    """
    """
    #[
    #
    if db is None:
        create_row = _create_row_no_value
    else:
        create_row = _create_row_with_value
    #
    all_rows = (
        create_row(k, date, action, status, db, )
        for k, v in register.items()
        for status, date in zip(v, base_span)
        if status is not None and status is not False
    )
    all_rows = sorted(all_rows, key=lambda row: (row[0], row[1], ), )
    row_groups = _it.groupby(all_rows, key=lambda row: (row[0], row[2], row[3], ), )
    for _, g in row_groups:
        representative = _create_representative(tuple(g), )
        table.add_row(representative, )
    #]


def _create_row_no_value(
    name: str,
    date: Period,
    action: str,
    status,
    *args,
) -> tuple:
    """
    """
    return (name, str(date), action, _get_status_symbol(status, ), )


def _create_row_with_value(
    name: str,
    date: Period,
    action: str,
    status,
    db: Databox,
) -> tuple:
    """
    """
    row = _create_row_no_value(name, date, action, status, )
    return row + (_get_value(db, name, date, ), )


def _get_status_symbol(status, ):
    """
    """
    return (
        status.symbol
        if hasattr(status, "symbol")
        else _PRETTY_SYMBOL.get(status, "")
    )


def _get_value(db, name, date, ):
    """
    """
    missing_str = Series._missing_str
    try:
        value = db[name][date][0, 0]
    except:
        return missing_str
    if _np.isnan(value):
        return missing_str
    return f"{value:g}"


def _create_representative(rows, ):
    if len(rows) == 1:
        return rows[0]
    else:
        return (rows[0][0], rows[0][1] + ">>" + rows[-1][1], rows[0][2], rows[0][3], )


_PRETTY_SYMBOL = {
    None: "",
    True: "⋅",
    False: "",
}


