"""
------------------------------------------------------------


Plans for dynamic simulations and steady state calculations
============================================================


------------------------------------------------------------
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Any, Protocol, NoReturn, )
from types import (EllipsisType, )
import warnings as _wa
import functools as _ft
import copy as _copy

from ..conveniences import copies as _copies
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from . import _pretty as _pretty
from . import _indexes as _indexes
from . import transforms as _transforms
#]


CHOOSE_TRANSFORM_CLASS = _transforms.CHOOSE_TRANSFORM_CLASS


__all__ = (
    "PlanSimulate",
    "PlanSteady",
    "Plan",
    "CHOOSE_TRANSFORM_CLASS",
)


class PlannableSimulateProtocol(Protocol, ):
    """
    """
    #[

    simulate_can_be_exogenized: Iterable[str] | None
    simulate_can_be_endogenized: Iterable[str] | None

    #]


class PlannableSteadyProtocol(Protocol, ):
    """
    """
    #[

    steady_can_be_exogenized: Iterable[str] | None
    steady_can_be_endogenized: Iterable[str] | None
    steady_can_be_fixed_level: Iterable[str] | None
    steady_can_be_fixed_change: Iterable[str] | None

    #]


class PlanSimulate(
    _pretty.PrettyMixin,
    _indexes.ItemMixin,
    _copies.CopyMixin,
):
    """
----------------------------------------------------------------------


Dynamic simulation plans
=========================

Dynamic simulation plans are used to set up assumptions about
[`Simultaneous`](../simultaneous_models/index.md) or
[`Sequential`](../sequential_models/index.md) model simulations:

* what variables to exogenize in what periods
* what shocks to endogenized in what periods (`Simultaneous` models only)

The plans only contain meta information, not any actual data. The actual
data (the exogenized data points, the values of shocks) need to be included
in the input databox.


----------------------------------------------------------------------
    """
    #[

    _registers = (
        "exogenized",
        "endogenized",
    )

    __slots__ = (
        *tuple(f"can_be_{r}" for r in _registers),
        *tuple(f"_{r}_register" for r in _registers),
        *tuple(f"default_{r}" for r in _registers),
        "base_span",
    )

    def _properties():
        """
------------------------------------------------------------


Directly accessible properties of `PlanSimulate` objects
=========================================================

Property | Description
---|---
`start_date` | Start date of the simulation span
`end_date` | End date of the simulation span
`num_periods` | Number of periods in the simulation span
`base_span` | Simulation span
`can_be_exogenized` | Names of quantities that can be exogenized
`can_be_endogenized` | Names of quantities that can be endogenized
`pretty` | Tabular view of the simulation plan

------------------------------------------------------------
        """

    def __init__(
        self,
        plannable: PlannableSimulateProtocol,
        span: Iterable[_dates.Dater] | None,
    ) -> None:
        """
        """
        self.base_span = tuple(span)
        self._default_exogenized = None
        self._default_endogenized = None
        for r in self._registers:
            register = {
                n: [None] * self.num_periods
                for n in getattr(plannable, f"simulate_can_be_{r}")
            } if hasattr(plannable, f"simulate_can_be_{r}") else {}
            setattr(self, f"can_be_{r}", tuple(register.keys()))
            setattr(self, f"_{r}_register", register)

    def check_consistency(
        self,
        plannable: PlannableSimulateProtocol,
        span: Iterable[_dates.Dater] | None,
        /,
    ) -> None:
        """
        """
        benchmark = type(self)(plannable, span, )
        if self.base_span != benchmark.base_span:
            raise _wrongdoings.IrisPieError(f"Plan span must be the same as the simulation span")
        for r in self._registers:
            if getattr(self, f"can_be_{r}") != getattr(benchmark, f"can_be_{r}"):
                raise _wrongdoings.IrisPieError(f"Plan must be created using the simulated model")

    @property
    def start_date(self, /, ) -> _dates.Dater:
        """
        """
        return self.base_span[0]

    @property
    def end_date(self, /, ) -> _dates.Dater:
        """
        """
        return self.base_span[-1]

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.base_span) if self.base_span is not None else 1

    @property
    def frequency(self, /, ) -> str:
        """
        Date frequency of simulation span
        """
        return self.start_date.frequency

    def exogenize(
        self,
        dates: Iterable[_dates.Dater] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
        *,
        transform: str | None = None,
        # when_data: bool | None = None,
        **kwargs,
    ) -> None:
        """
------------------------------------------------------------


`exogenize`
===========

#### Exogenize certain quantities at certain dates ####

Syntax
-------

    self.exogenize(dates, names, **options )

Input arguments
----------------

### `dates` ###
Dates at which the `names` will be exogenized; use `...` for all simulation dates.

### `names` ###
Names of quantities to exogenize at the `dates`; use `...` for all exogenizable quantities.

Options
--------

### `transform=None` ###
Transformation (a string) to be applied to the exogenized quantities; only
available in simulation plans created for
[`Sequential`](../Sequential/index.md) objects.


------------------------------------------------------------
        """
        transform = _transforms.resolve_transform(transform, **kwargs, )
        self._plan_simulate(self._exogenized_register, dates, names, transform, )

    def get_names_exogenized_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
        """
        return self._get_names_registered_in_period(self._exogenized_register, *args, **kwargs, )

    def endogenize(
        self,
        dates: Iterable[_dates.Dater] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
    ) -> None:
        """
------------------------------------------------------------

# `endogenize`

#### Endogenize certain quantities at certain dates ####

Syntax
-------


------------------------------------------------------------
        """
        transform = None
        new_status = True
        self._plan_simulate(self._endogenized_register, dates, names, new_status, )

    def get_names_endogenized_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
        """
        return self._get_names_registered_in_period(self._endogenized_register, *args, **kwargs, )

    def swap(
        self,
        dates: Iterable[_dates.Dater] | EllipsisType,
        pairs: Iterable[tuple[str, str]] | tuple[str, str],
        *args, **kwargs,
    ) -> None:
        """
        """
        pairs = tuple(pairs)
        if not pairs:
            return
        if len(pairs) == 2 and isinstance(pairs[0], str) and isinstance(pairs[1], str):
            pairs = (pairs, )
        for pair in pairs:
            self.exogenize(dates, pair[0], *args, **kwargs, )
            self.endogenize(dates, pair[1], *args, **kwargs, )

    def get_exogenized_point(
        self,
        name: str,
        column: int,
        /,
    ) -> _transforms.PlanTransformProtocol | None:
        """
        """
        point = self._exogenized_register[name][column]
        return point if point is not None else self._default_exogenized

    def get_endogenized_point(
        self,
        name: str,
        column: int,
        /,
    ) -> _transforms.PlanTransformProtocol | None:
        """
        """
        point = self._endogenized_register[name][column]
        return point if point is not None else self._default_endogenized

    def get_databox_names(self, /, ) -> tuple[str]:
        """
        """
        databox_names = set()
        for k, v in self._exogenized_register.items():
            databox_names.update(
                t.resolve_databox_name(k, )
                for t in v if t is not None
            )
        return tuple(n for n in databox_names if n is not None)

    def _get_names_registered_in_period(
        self,
        register: dict[str, Any],
        date: _dates.Dater,
    ) -> tuple[str, ...]:
        """
        """
        column_index = self.base_span.index(date)
        return tuple(
            name
            for name, status in register.items()
            if bool(status[column_index])
        )

    def __str__(self, /, ) -> str:
        """
        """
        return self.get_pretty_string()

    def _plan_simulate(
        self,
        register: dict,
        dates: Iterable[_dates.Dater] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        new_status: Any,
    ) -> None:
        """
        """
        names = _resolve_and_check_names(register, names, )
        date_indices = self._get_date_indices(dates, )
        for n in names:
            for t in date_indices:
                register[n][t] = new_status

    def _get_date_indices(
        self,
        dates: Iterable[_dates.Dater] | EllipsisType,
        /,
    ) -> tuple[int, ...]:
        """
        """
        if dates is Ellipsis:
            return tuple(range(len(self.base_span)))
        dates = dates.resolve(self, ) if hasattr(dates, "resolve") else dates
        catch_invalid_dates(dates, self.base_span, )
        return tuple(d - self.start_date for d in dates)

    #]


Plan = PlanSimulate


class PlanSteady():
    """
------------------------------------------------------------


`PlanSteady`
============

#### Plan for steady state calculations ####


------------------------------------------------------------
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
        plannable: PlannableSteadyProtocol,
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
    names: Iterable[str] | str | EllipsisType,
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


def catch_invalid_dates(
    dates: Iterable[_dates.Dater],
    base_span: tuple[_dates.Dater],
    /,
) -> NoReturn | None:
    """
    """
    invalid = [repr(d) for d in dates if d not in base_span]
    if invalid:
        raise _wrongdoings.IrisPieError(
            ["These date(s) are out of simulation span:"] + invalid
        )

