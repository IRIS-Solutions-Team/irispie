"""
Simulation plans
"""


#[
from __future__ import annotations

from typing import (Iterable, Protocol, )
import warnings as _wa
import prettytable as _pt

from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from .. import transforms as _transforms
#]


__all__ = (
    "Plan",
)

class PlannableProtocol(Protocol, ):
    """
    """
    #[

    can_be_exogenized: Iterable[str] | None
    can_be_endogenized: Iterable[str] | None
    can_be_fixed_level: Iterable[str] | None
    can_be_fixed_change: Iterable[str] | None

    #]


class GetPlannableForSimulateProtocol(Protocol, ):
    """
    """
    #[

    def get_plannable_for_simulate(self, ) -> PlannableForSimulate: ...

    #]


class GetPlannableForSteadyProtocol(Protocol, ):
    """
    """
    #[

    def get_plannable_for_steady(self, ) -> PlannableForSimulate: ...

    #]


class Plan:
    """
    """
    #[

    __slots__ = (
        "base_range",
        "anticipate",
        "exogenized",
        "endogenized",
        "fixed_level",
        "fixed_change",
    )

    def __init__(
        self,
        plannable: GetPlannableProtocol,
        base_range: Iterable[Dater] | None,
        /,
        anticipate: bool | None = True,
    ) -> None:
        """
        """
        self.base_range = \
            tuple(base_range) \
            if base_range is not None \
            else None
        self.anticipate = anticipate
        #
        if hasattr(plannable, "get_plannable_for_simulate"):
            _wa.warn("Warning: Use Plan.for_simulation() instead of Plan()", SyntaxWarning, )
            plannable = plannable.get_plannable_for_simulate()
        #
        self._initialize(plannable, )

    @classmethod
    def for_simulate(
        cls,
        model: GetPlannableForSimulateProtocol,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
    ) -> Self:
        """
        """
        plannable = model.get_plannable_for_simulate()
        return cls(plannable, base_range, anticipate=anticipate, )

    @classmethod
    def for_steady(
        cls,
        model: GetPlannableForSteadyProtocol,
        /,
    ) -> Self:
        """
        """
        plannable = model.get_plannable_for_steady()
        return cls(plannable, None, anticipate=None, )

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

    @property
    def can_be_exogenized(self, /, ) -> tuple[str]:
        """
        """
        return tuple(self.exogenized.keys())

    @property
    def can_be_endogenized(self, /, ) -> tuple[str]:
        """
        """
        return tuple(self.endogenized.keys())

    def _initialize(
        self,
        plannable: PlannableProtocol,
        /,
    ) -> None:
        """
        """
        #
        # REFACTOR
        # as descriptors
        #
        self.exogenized = {
            n: [None] * self.num_periods
            for n in plannable.can_be_exogenized
        } if plannable.can_be_exogenized else {}
        #
        self.endogenized = {
            n: [None] * self.num_periods
            for n in plannable.can_be_endogenized
        } if plannable.can_be_endogenized else {}
        #
        self.fixed_level = {
            n: [None] * self.num_periods
            for n in plannable.can_be_fixed_level
        } if plannable.can_be_fixed_level else {}
        #
        self.fixed_change = {
            n: [None] * self.num_periods
            for n in plannable.can_be_fixed_change
        } if plannable.can_be_fixed_change else {}

    def exogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        /,
        transform: str | None = None,
        when_data: bool = False,
    ) -> None:
        """
        """
        names = self._resolve_and_check_names(self.exogenized, names, "exogenized", )
        date_indexes = self._resolve_dates(dates, )
        transform = _transforms.RESOLVE_TRANSFORM[transform](when_data, )
        for n in names:
            for t in date_indexes:
                self.exogenized[n][t] = transform

    def collect_databank_names(self, /, ) -> tuple[str]:
        """
        """
        databank_names = set()
        for k, v in self.exogenized.items():
            databank_names.update(
                t.resolve_databank_name(k, )
                for t in v if t is not None
            )
        return tuple(databank_names)

    def _resolve_and_check_names(
        self,
        what: dict,
        names: Iterable[str] | str | Ellipsis,
        done: str,
        /,
    ) -> tuple[str]:
        """
        """
        if names is Ellipsis:
            return tuple(what.keys())
        if isinstance(names, str):
            names = (names, )
        names = tuple(names)
        invalid = [n for n in names if n not in what]
        if invalid:
            raise _wrongdoings.IrisPieError(
                [f"These name(s) cannot be {done}:"] + invalid
            )
        return names

    def _resolve_dates(
        self,
        dates,
        /,
    ) -> tuple[int, ...]:
        """
        """
        if dates is Ellipsis:
            return tuple(range(self.num_periods))
        invalid = [repr(d) for d in dates if d not in self.base_range]
        if invalid:
            raise _wrongdoings.IrisPieError(
                ["These date(s) are out of simulation range:"] + invalid
            )
        return tuple(d - self.start_date for d in dates)

    def get_pretty_table(self, *args, **kwargs, ) -> _pt.PrettyTable:
        """
        """
        table = _pt.PrettyTable()
        table.align = "r"
        table.field_names = ("", ) + tuple("{:>10}".format(table) for table in self.base_range)
        for name in self.can_be_exogenized:
            if _is_pristine(self.exogenized[name], ):
                continue
            table.add_row([name] + [ (str(i) if i is not None else "") for i in self.exogenized[name] ], )
        return table

    def get_pretty_string(self, *args, **kwargs, ) -> str:
        """
        """
        return self.get_pretty_table(*args, **kwargs, ).get_string()

    def __str__(self, /, ) -> str:
        """
        """
        return self.get_pretty_string()

    #]


Plan.for_simulation = Plan.for_simulate


def _is_pristine(row: tuple, /, ) -> bool:
    """
    """
    return all(i is None for i in row)

