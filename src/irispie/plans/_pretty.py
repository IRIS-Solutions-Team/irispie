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
    ) -> _pt.PrettyTable:
        """
        """
        table = _pt.PrettyTable()
        table.field_names = ("NAME", "PERIOD", "REGISTER", "TRANSFORM", )
        table.align = "r"
        table.align["NAME"] = "l"
        for r in ("exogenized", "endogenized", ):
            if getattr(self, f"_{r}_register"):
                _add_register_to_table(
                    table,
                    self.base_span,
                    getattr(self, f"_{r}_register"),
                    r,
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
    base_span,
    register,
    action,
) -> None:
    """
    """
    #[
    for k, v in register.items():
        for status, date in zip(v, base_span):
            if not status:
                continue
            symbol = status.symbol if hasattr(status, "symbol") else _PRETTY_SYMBOL.get(status, "")
            table.add_row((k, str(date), action, symbol, ))
    #]


_PRETTY_SYMBOL = {
    None: "",
    True: "*",
    False: "",
}

