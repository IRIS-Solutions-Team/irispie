"""
Simulation plans
"""


#[
from __future__ import annotations

from typing import (Iterable, Protocol, )

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
    def can_be_exogenized(self, /, ) -> Iterable[str]: ...
    def can_be_endogenized(self, /, ) -> Iterable[str]: ...
    #]


class Plan:
    """
    """
    __slots__ = (
        "base_range",
        "anticipate",
        "exogenized",
        "endogenized",
    )

    def __init__(
        self,
        plannable,
        base_range: Iterable[Dater],
        /,
        anticipate: bool = True,
    ) -> None:
        """
        """
        self.base_range = tuple(base_range)
        self.anticipate = anticipate
        self._initialize(plannable, )

    @property
    def start_date(self, /, ) -> Dater:
        """
        """
        return self.base_range[0]

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.base_range)

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
        can_be_exogenized = plannable.can_be_exogenized()
        self.exogenized = {
            n: [None] * self.num_periods
            for n in can_be_exogenized
        } if can_be_exogenized else {}
        #
        can_be_endogenized = plannable.can_be_endogenized()
        self.endogenized = {
            n: [None] * self.num_periods
            for n in can_be_endogenized
        } if can_be_endogenized else {}

    def exogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str,
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
        names: Iterable[str] | str,
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

