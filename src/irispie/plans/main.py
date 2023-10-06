"""
Simulation plans
"""


#[
from __future__ import annotations

from typing import (Iterable, Protocol, )
import warnings as _wa
import prettytable as _pt
import functools as _ft

from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import transforms as _transforms
#]


__all__ = (
    "SimulatePlan", "Plan", "SteadyPlan",
)


class SimulatePlannableProtocol(Protocol, ):
    """
    """
    #[

    simulate_can_be_exogenized: Iterable[str] | None
    simulate_can_be_endogenized: Iterable[str] | None

    #]


class SteadyPlannableProtocol(Protocol, ):
    """
    """
    #[

    steady_can_be_exogenized: Iterable[str] | None
    steady_can_be_endogenized: Iterable[str] | None
    steady_can_be_fixed_level: Iterable[str] | None
    steady_can_be_fixed_change: Iterable[str] | None

    #]


class SimulatePlan:
    """
    """
    #[

    __slots__ = (
        "base_range",
        "anticipate",
        "can_be_exogenized",
        "can_be_endogenized",
        "_exogenized_register",
        "_endogenized_register",
    )

    def __init__(
        self,
        plannable: SimulatePlannableProtocol,
        base_range: Iterable[Dater] | None,
        /,
        anticipate: bool = True,
    ) -> None:
        """
        """
        self.base_range = tuple(base_range)
        self.anticipate = bool(anticipate)
        #
        for r in ("exogenized", "endogenized", ):
            register = {
                n: [None] * self.num_periods
                for n in getattr(plannable, f"simulate_can_be_{r}")
            } if hasattr(plannable, f"simulate_can_be_{r}") else {}
            setattr(self, f"can_be_{r}", tuple(register.keys()))
            setattr(self, f"_{r}_register", register)

    @property
    def start_date(self, /, ) -> Dater:
        """
        """
        return self.base_range[0]

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.base_range) if self.base_range is not None else 1

    def exogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        /,
        *,
        anticipate: bool | None = None,
        transform: str | None = None,
        when_data: bool | None = False,
    ) -> None:
        """
        """
        anticipate = self.anticipate if anticipate is None else bool(anticipate)
        _plan_simulate(
            self._exogenized_register,
            self.base_range,
            dates,
            names,
            anticipate,
            transform,
            when_data,
        )

    def endogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        /,
        *,
        anticipate: bool | None = None,
    ) -> None:
        """
        """
        transform = None
        when_data = None
        anticipate = self.anticipate if anticipate is None else bool(anticipate)
        _plan_simulate(
            self._endogenized_register,
            self.base_range,
            dates,
            names,
            anticipate,
            transform,
            when_data,
        )

    def get_exogenized_point(
        self,
        name: str,
        column: int,
        /,
    ) -> _transforms.Transform | None:
        """
        """
        return self._exogenized_register[name][column]

    def get_endogenized_point(
        self,
        name: str,
        column: int,
        /,
    ) -> _transforms.Transform | None:
        """
        """
        return self._endogenized_register[name][column]

    def collect_databox_names(self, /, ) -> tuple[str]:
        """
        """
        databox_names = set()
        for k, v in self._exogenized_register.items():
            databox_names.update(
                t.resolve_databox_name(k, )
                for t in v if t is not None
            )
        return tuple(databox_names)

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
        Create pretty table for the Plan
        """
        table = _pt.PrettyTable()
        table.field_names = ("", ) + tuple("{:>10}".format(table) for table in self.base_range)
        table.align = "r"
        table.align[""] = "l"
        if hasattr(self, "_exogenized_register"):
            _add_register_to_table(table, self._exogenized_register, full, )
        if hasattr(self, "_endogenized_register"):
            _add_register_to_table(table, self._endogenized_register, full, )
        return table

    def pretty_print(self, *args, **kwargs, ) -> None:
        """
        """
        print(self.get_pretty_table(*args, **kwargs, ), )

    def get_pretty_string(self, *args, **kwargs, ) -> str:
        """
        """
        return self.get_pretty_table(*args, **kwargs, ).get_string()

    def __str__(self, /, ) -> str:
        """
        """
        return self.get_pretty_string()

    #]


Plan = SimulatePlan


def _add_register_to_table(
    table,
    register,
    full,
) -> None:
    """
    """
    #[
    previous = None
    for k, v in register.items():
        if full or _has_point(v, ):
            dates = [ (str(i) if i is not None else "") for i in v ]
            if previous:
                table.add_row(previous, )
            previous = [k] + dates
    if previous:
        table.add_row(previous, divider=True, )
    #]


def _has_point(row: list, /, ) -> bool:
    """
    """
    return any(i is not None for i in row)


class SteadyPlan:
    """
    """
    #[

    __slots__ = (
        "can_be_exogenized",
        "can_be_endogenized",
        "can_be_fixed_level",
        "can_be_fixed_change",
        "_exogenized_register",
        "_endogenized_register",
        "_fixed_level_register",
        "_fixed_change_register",
    )

    def __init__(
        self,
        plannable: SteadyPlannableProtocol,
        /,
    ) -> None:
        """
        """
        for r in ("exogenized", "endogenized", "fixed_level", "fixed_change", ):
            register = {
                n: False
                for n in getattr(plannable, f"steady_can_be_{r}")
            } if hasattr(plannable, f"steady_can_be_{r}") else {}
            setattr(self, f"can_be_{r}", tuple(register.keys()))
            setattr(self, f"_{r}_register", register)

    #]

#         self.exogenized = {
#             n: [None] * self.num_periods
#             for n in plannable.can_be_exogenized
#         } if plannable.can_be_exogenized else {}
#         #
#         self.endogenized = {
#             n: [None] * self.num_periods
#             for n in plannable.can_be_endogenized
#         } if plannable.can_be_endogenized else {}
#         #
#         self.fixed_level = {
#             n: [None] * self.num_periods
#             for n in plannable.can_be_fixed_level
#         } if plannable.can_be_fixed_level else {}
#         #
#         self.fixed_change = {
#             n: [None] * self.num_periods
#             for n in plannable.can_be_fixed_change
#         } if plannable.can_be_fixed_change else {}


def _resolve_and_check_names(
    register: dict | None,
    names: Iterable[str] | str | Ellipsis,
    /,
) -> tuple[str]:
    """
    """
    keys = tuple(register.keys()) if register else ()
    if names is Ellipsis:
        return keys
    if isinstance(names, str):
        names = (names, )
    names = tuple(names)
    invalid = [n for n in names if n not in keys]
    if invalid:
        raise _wrongdoings.IrisPieError(
            [f"Invalid names:"] + invalid
        )
    return names


def _resolve_dates(
    base_range: Iterable[_dates.Dater],
    dates: Iterable[_dates.Dater] | Ellipsis,
    /,
) -> tuple[int, ...]:
    """
    """
    if dates is Ellipsis:
        return tuple(range(len(base_range)))
    invalid = [repr(d) for d in dates if d not in base_range]
    if invalid:
        raise _wrongdoings.IrisPieError(
            ["These date(s) are out of simulation range:"] + invalid
        )
    return tuple(d - base_range[0] for d in dates)


def _plan_simulate(
    register: dict,
    base_range: Iterable[_dates.Dater],
    dates: Iterable[_dates.Dater] | Ellipsis,
    names: Iterable[str] | str | Ellipsis,
    anticipate: bool | None,
    transform: str | None,
    when_data: bool | None,
) -> None:
    """
    """
    names = _resolve_and_check_names(register, names, )
    date_indexes = _resolve_dates(base_range, dates, )
    transform = _transforms.RESOLVE_TRANSFORM[transform](anticipate, when_data, )
    for n in names:
        for t in date_indexes:
            register[n][t] = transform

