"""
"""


#[
from __future__ import annotations

from typing import (Self, Any, )
import prettytable as _pt
#]


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
        full: bool = False,
    ) -> _pt.PrettyTable:
        """
        """
        table = _pt.PrettyTable()
        table.field_names = ("", ) + tuple("{:>10}".format(table) for table in self.base_range)
        table.align = "r"
        table.align[""] = "l"
        for r in ("exogenized", "endogenized", "anticipated", ):
            if getattr(self, f"_{r}_register"):
                _add_register_to_table(
                    table,
                    getattr(self, f"_{r}_register"),
                    getattr(self, f"_default_{r}"),
                    full,
                )
        return table

    def pretty_print(self, *args, **kwargs, ) -> None:
        """
        """
        print(self.get_pretty_table(*args, **kwargs, ), )

    def get_pretty_string(self, *args, **kwargs, ) -> str:
        """
        """
        return self.get_pretty_table(*args, **kwargs, ).get_string()


def _add_register_to_table(
    table,
    register,
    default,
    full,
) -> None:
    """
    """
    #[
    previous = None
    for k, v in register.items():
        if full or any(v):
            points = [ _PRETTY_SYMBOL.get(i, _PRETTY_SYMBOL[Any]) for i in v ]
            if previous:
                table.add_row(previous, )
            previous = [k] + points
    if previous:
        table.add_row(previous, divider=True, )
    #]


_PRETTY_SYMBOL = {
    None: "",
    True: "T",
    False: "F",
    Any: "◼︎",
}

