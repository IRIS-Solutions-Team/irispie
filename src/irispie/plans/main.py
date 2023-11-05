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
import warnings as _wa
import functools as _ft
import copy as _copy

from ..conveniences import copies as _copies
from .. import dates as _dates
from .. import wrongdoings as _wrongdoings
from . import _pretty as _pretty
from . import _indexes as _indexes
from . import _transforms as _transforms
#]


PlanTransformFactory = _transforms.PlanTransformFactory


__all__ = (
    "PlanSimulate",
    "PlanSteady",
    "Plan",
    "PlanTransformFactory",
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
------------------------------------------------------------


Plans for dynamic simulations
===============================


------------------------------------------------------------
    """
    #[

    __slots__ = (
        "base_range",
        "can_be_anticipated",
        "can_be_exogenized",
        "can_be_endogenized",
        "_exogenized_register",
        "_endogenized_register",
        "_anticipated_register",
        "_default_exogenized",
        "_default_endogenized",
        "_default_anticipated",
    )

    def properties():
        """
------------------------------------------------------------


Properties of `PlanSimulate` objects
=====================================

#### `start_date` ####
Start date of the simulation range

#### `num_periods` ####
Number of periods in the simulation range

#### `base_range` ####
Simulation range

#### `anticipate` ####
Default anticipation status

#### `can_be_exogenized` ####
Names of quantities that can be exogenized

#### `can_be_endogenized` ####
Names of quantities that can be endogenized

#### `can_be_anticipated` ####
Names of quantities that have anticipation status

#### `pretty` ####
Tabular view of the simulation plan


------------------------------------------------------------
        """

    #[

    def __init__(
        self,
        plannable: PlannableSimulateProtocol,
        base_range: Iterable[_dates.Dater] | None,
        /,
        anticipate: bool = True,
    ) -> None:
        """
        """
        self.base_range = tuple(base_range)
        self._default_exogenized = None
        self._default_endogenized = None
        self._default_anticipated = bool(anticipate)
        for r in ("exogenized", "endogenized", "anticipated", ):
            register = {
                n: [None] * self.num_periods
                for n in getattr(plannable, f"simulate_can_be_{r}")
            } if hasattr(plannable, f"simulate_can_be_{r}") else {}
            setattr(self, f"can_be_{r}", tuple(register.keys()))
            setattr(self, f"_{r}_register", register)

    @property
    def start_date(self, /, ) -> _dates.Dater:
        """
        """
        return self.base_range[0]

    @property
    def end_date(self, /, ) -> _dates.Dater:
        """
        """
        return self.base_range[-1]

    @property
    def num_periods(self, /, ) -> int:
        """
        """
        return len(self.base_range) if self.base_range is not None else 1

    def anticipate(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        /,
        new_status: bool | None = None,
    ) -> None:
        self._plan_simulate(
            self._anticipated_register,
            dates,
            names,
            new_status,
        )

    def exogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        /,
        *,
        transform: str | None = None,
        when_data: bool | None = False,
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

### `when_data=False` ###
Exogenize the quantities at the dates when the data for the quantities are
available; only available in simulation plans created for
[`Sequential`](../Sequential/index.md) objects.


------------------------------------------------------------
        """
        transform = _transforms.resolve_transform(transform)
        transform.when_data = when_data
        self._plan_simulate(
            self._exogenized_register,
            dates,
            names,
            transform,
        )

    def endogenize(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
    ) -> None:
        """

        Endogenize

        """
        transform = None
        when_data = None
        new_status = True
        self._plan_simulate(
            self._endogenized_register,
            dates,
            names,
            new_status,
        )

    def get_anticipated_point(
        self,
        name: str,
        column: int,
        /,
    ) -> bool | None:
        """
        """
        point = self._anticipated_register[name][column]
        return point if point is not None else self._default_anticipated

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

    def __str__(self, /, ) -> str:
        """
        """
        return self.get_pretty_string()

    def _plan_simulate(
        self,
        register: dict,
        dates: Iterable[_dates.Dater] | Ellipsis,
        names: Iterable[str] | str | Ellipsis,
        new_status: Any,
    ) -> None:
        """
        """
        anticipate = True
        names = _resolve_and_check_names(register, names, )
        date_indices = self._get_date_indices(dates, )
        for n in names:
            for t in date_indices:
                register[n][t] = new_status

    def _get_date_indices(
        self,
        dates: Iterable[_dates.Dater] | Ellipsis,
        /,
    ) -> tuple[int, ...]:
        """
        """
        if dates is Ellipsis:
            return tuple(range(len(self.base_range)))
        dates = dates.resolve(self, ) if hasattr(dates, "resolve") else dates
        catch_invalid_dates(dates, self.base_range, )
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


def catch_invalid_dates(
    dates: Iterable[_dates.Dater],
    base_range: tuple[_dates.Dater],
    /,
) -> NoReturn | None:
    """
    """
    invalid = [repr(d) for d in dates if d not in base_range]
    if invalid:
        raise _wrongdoings.IrisPieError(
            ["These date(s) are out of simulation range:"] + invalid
        )

