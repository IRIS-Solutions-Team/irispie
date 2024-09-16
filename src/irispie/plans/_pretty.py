"""
"""


#[

from __future__ import annotations

from typing import (Self, Any, )
import numpy as _np
import prettytable as _pt
import documark as _dm

from ..databoxes.main import (Databox, )
from ..dates import (Period, )

#]



class Mixin:
    """
    """
    #[

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

    def get_table(self, /, *args, **kwargs, ) -> _pt.PrettyTable:
        """
        """
        table = _pt.PrettyTable()
        table.field_names = self._TABLE_FIELDS
        table.align = "r"
        table.align["NAME"] = "l"
        for action in self._registers:
            register = self.get_register_by_name(action, )
            if register:
                self._add_register_to_table(
                    table, register, action,
                    *args, **kwargs,
                )
        return table

    get_pretty = get_table

    @_dm.reference(
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

