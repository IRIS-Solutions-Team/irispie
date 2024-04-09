"""
Plans for dynamic simulations and steady state calculations
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Any, Protocol, NoReturn, )
from types import (EllipsisType, )
import warnings as _wa
import functools as _ft
import numpy as _np
import copy as _copy

from ..conveniences import copies as _copies
from ..dates import (Period, )
from .. import wrongdoings as _wrongdoings
from .. import pages as _pages
from . import _pretty as _pretty
from . import _indexes as _indexes
from . import transforms as _transforms
#]


CHOOSE_TRANSFORM_CLASS = _transforms.CHOOSE_TRANSFORM_CLASS


__all__ = (
    "SimulationPlan", "PlanSimulate", "Plan",
    "PlanSteady", "SteadyPlan",
    "CHOOSE_TRANSFORM_CLASS",
)


class SimulationPlannableProtocol(Protocol, ):
    """
    """
    #[

    simulate_can_be_exogenized: Iterable[str] | None
    simulate_can_be_endogenized: Iterable[str] | None
    simulate_can_be_exogenized_anticipated: Iterable[str] | None
    simulate_can_be_exogenized_anticipated: Iterable[str] | None
    simulate_can_be_endogenized_unanticipated: Iterable[str] | None
    simulate_can_be_endogenized_unanticipated: Iterable[str] | None

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


@_pages.reference(
    path=("structural_models", "simulation_plans.md", ),
    categories={
        "constructor": "Creating new simulation plans",
        "property": None,
        "definition": "Defining exogenized and endogenized data points",
        "information": "Getting information about simulation plans",
    },
)
class SimulationPlan(
    _pretty.PrettyMixin,
    _indexes.ItemMixin,
    _copies.CopyMixin,
):
    """
················································································

Simulation plans
=================

`SimulationPlan` objects are used to set up conditioning assumptions
for simulating [`Simultaneous`](simultaneous_modelsd) or
[`Sequential`](sequential_models) models. The simulation plans specify

* what variables to exogenize in what periods
* what shocks to endogenized in what periods (`Simultaneous` models only)

The plans only contain meta information, not the actual data points for the
exogenized variables. The actual data points are included in the input databox.

················································································
    """
    #[

    _registers = (
        "exogenized",
        "endogenized",
        "exogenized_anticipated",
        "exogenized_unanticipated",
        "endogenized_unanticipated",
        "endogenized_anticipated",
    )

    __slots__ = (
        *tuple(f"can_be_{r}" for r in _registers),
        *tuple(f"_{r}_register" for r in _registers),
        *tuple(f"default_{r}" for r in _registers),
        "base_span",
    )

    def _properties():
        """
················································································


Directly accessible properties of `SimulationPlan` objects
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

················································································
        """

    @_pages.reference(category="constructor", call_name="SimulationPlan", )
    def __init__(
        self,
        model,
        span: Iterable[Period] | None,
    ) -> None:
        """
················································································

==Create new simulation plan object==

```
self = SimulationPlan(model, time_span, )
```

Create a new simulation plan object for a
[`Simultaneous`](sequential_models) or
[`Sequential`](sequential_models) model.

### Input arguments ###

???+ input "model"

    A [`Simultaneous`](sequential_models) or
    [`Sequential`](sequential_models) model that will be simulated.

???+ input "time_span"

    A date range on which the `model` will be simulated.


### Returns ###

???+ returns "self"

    A new empty simulation plan object.

················································································
        """
        self.base_span = tuple(span)
        self._default_exogenized = None
        self._default_endogenized = None
        plannable = model.get_simulation_plannable()
        for r in self._registers:
            can_be_name = f"can_be_{r}"
            register = {
                n: [None] * self.num_periods
                for n in getattr(plannable, can_be_name, )
            } if hasattr(plannable, can_be_name, ) else {}
            setattr(self, can_be_name, tuple(register.keys()))
            setattr(self, f"_{r}_register", register)

    def check_consistency(
        self,
        plannable: SimulationPlannableProtocol,
        span: Iterable[Period] | None,
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
    @_pages.reference(category="property", )
    def start_date(self, /, ) -> Period:
        """==Start date of the simulation span=="""
        return self.base_span[0]

    @property
    @_pages.reference(category="property", )
    def end_date(self, /, ) -> Period:
        """==End date of the simulation span=="""
        return self.base_span[-1]

    @property
    @_pages.reference(category="property", )
    def num_periods(self, /, ) -> int:
        """==Number of periods in the simulation span=="""
        return len(self.base_span) if self.base_span is not None else 1

    @property
    @_pages.reference(category="property", )
    def frequency(self, /, ) -> str:
        """==Date frequency of the simulation span=="""
        return self.start_date.frequency

    @_pages.reference(category="definition", )
    def exogenize(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
        *,
        transform: str | None = None,
        # when_data: bool | None = None,
        **kwargs,
    ) -> None:
        """
················································································

==Exogenize certain quantities at certain dates==

```
self.exogenize(
    dates, names,
    /,
    transform=None,
    when_data=False,
)
```

### Input arguments ###


???+ input "dates"

    Dates at which the `names` will be exogenized; use `...` for all simulation dates.

???+ input "names"

    Names of quantities to exogenize at the `dates`; use `...` for all exogenizable quantities.


### Input arguments available only for `Sequential` models ###

???+ input "transform"

    Transformation (specified as a string) to be applied to the exogenized
    quantities; if `None`, no tranformation is applied.

???+ input "when_data"

    If `True`, the data point will be exogenized only if a proper value
    exists in the input data.

················································································
        """
        transform = _transforms.resolve_transform(transform, **kwargs, )
        self._register(self._exogenized_register, dates, names, transform, )

    @_pages.reference(category="definition", )
    def exogenize_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
    ) -> None:
        """
················································································

==Exogenize certain quantities at certain dates==

```
self.exogenize_anticipated(
    dates,
    names,
)
```

### Input arguments ###


???+ input "dates"

    Dates at which the `names` will be exogenized; use `...` for all simulation dates.

???+ input "names"

    Names of quantities to exogenize at the `dates`; use `...` for all exogenizable quantities.

················································································
        """
        self._register(self._exogenized_anticipated_register, dates, names, True, )

    @_pages.reference(category="definition", )
    def exogenize_unanticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
    ) -> None:
        """
················································································

==Exogenize certain quantities at certain dates==

```
self.exogenize_unanticipated(
    dates, names,
    /,
    transform=None,
    when_data=False,
)
```

### Input arguments ###


???+ input "dates"

    Dates at which the `names` will be exogenized; use `...` for all simulation dates.

???+ input "names"

    Names of quantities to exogenize at the `dates`; use `...` for all exogenizable quantities.


### Input arguments available only for `Sequential` models ###

???+ input "transform"

    Transformation (specified as a string) to be applied to the exogenized
    quantities; if `None`, no tranformation is applied.

???+ input "when_data"

    If `True`, the data point will be exogenized only if a proper value
    exists in the input data.

················································································
        """
        self._register(self._exogenized_unanticipated_register, dates, names, True, )

    @_pages.reference(category="information", )
    def get_names_exogenized_unanticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names exogenized at a certain date==

················································································
        """
        return self._get_names_registered_in_period(
            self._exogenized_unanticipated_register,
            *args, **kwargs,
        )

    @_pages.reference(category="definition", )
    def endogenize(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
    ) -> None:
        """
················································································

==Endogenize certain quantities at certain dates==

················································································
        """
        self._register(self._endogenized_register, dates, names, True, )

    @_pages.reference(category="definition", )
    def endogenize_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
    ) -> None:
        """
················································································

==Endogenize certain quantities at certain dates==

················································································
        """
        self._register(self._endogenized_anticipated_register, dates, names, True, )

    @_pages.reference(category="definition", )
    def endogenize_unanticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
    ) -> None:
        """
················································································

==Endogenize certain quantities at certain dates==

················································································
        """
        self._register(self._endogenized_unanticipated_register, dates, names, True, )

    @_pages.reference(category="information", )
    def get_names_endogenized_unanticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names endogenized at a certain date==

················································································
        """
        return self._get_names_registered_in_period(
            self._endogenized_unanticipated_register,
            *args,
            **kwargs,
        )

    def swap(
        self,
        dates: Iterable[Period] | EllipsisType,
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

    def swap_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
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
            self.exogenize_anticipated(dates, pair[0], *args, **kwargs, )
            self.endogenize_anticipated(dates, pair[1], *args, **kwargs, )

    def swap_unanticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
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
            self.exogenize_unanticipated(dates, pair[0], *args, **kwargs, )
            self.endogenize_unanticipated(dates, pair[1], *args, **kwargs, )

    def tabulate_registered_points(
        self: Self,
        register_name: str,
        names: str | Iterable[str] | EllipsisType,
        periods: Iterable[Period] | EllipsisType,
    ) -> _np.ndarray | None:
        """
        """
        per_indexes = self._get_period_indexes(periods, )
        register = getattr(self, f"_{register_name}_register", )
        names = _resolve_and_check_names(register, names, )
        num_pers = len(per_indexes)
        num_names = len(names)
        return _np.array(tuple(
            tuple( bool(register[n][t]) for t in per_indexes )
            for n in names
        )) if names and per_indexes else None

    def get_exogenized_point(
        self,
        name: str,
        date: Period,
        /,
    ) -> _transforms.PlanTransformProtocol | None:
        """
        """
        column = next((
            column
            for column, t in enumerate(self.base_span)
            if date == t
        ))
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
        date: Period,
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

    def _register(
        self,
        register: dict,
        periods: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        new_status: Any,
    ) -> None:
        """
        """
        names = _resolve_and_check_names(register, names, )
        per_indices = self._get_period_indexes(periods, )
        for n in names:
            for t in per_indices:
                register[n][t] = new_status

    def _get_period_indexes(
        self,
        dates: Iterable[Period] | EllipsisType,
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


@_pages.reference(
    path=("structural_models", "steady_plans.md", ),
    categories={
        "constructor": "Creating new steady plans",
        "property": None,
        "definition": "Defining exogenized, endogenized and fixed quantities",
    },
)
class SteadyPlan:
    """
················································································

Steady-state calculation plans
===============================

`SteadyPlan` objects are used to define assumptions about the steady state
values of certain model variables.

················································································
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

    @_pages.reference(category="constructor", call_name="SteadyPlan", )
    def __init__(
        self,
        model,
        /,
    ) -> None:
        """
················································································

==Create new steady plan object==

················································································
        """
        plannable = model.get_steady_plannable()
        for r in ("exogenized", "endogenized", "fixed_level", "fixed_change", ):
            can_be_name = f"can_be_{r}"
            register = {
                n: False
                for n in getattr(plannable, can_be_name, )
            } if hasattr(plannable, can_be_name, ) else {}
            setattr(self, can_be_name, tuple(register.keys()))
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

    @_pages.reference(category="definition", )
    def exogenize(
        self,
        names: Iterable[str] | str,
        /,
    ) -> None:
        """
················································································

==Exogenize steady levels of variables==

················································································
        """
        pass

    @_pages.reference(category="definition", )
    def endogenize(
        self,
        names: Iterable[str] | str,
        /,
    ) -> None:
        """
················································································

==Endogenize parameters==

················································································
        """
        pass

    @_pages.reference(category="definition", )
    def fix_level(
        self,
        names: Iterable[str] | str,
        /,
    ) -> None:
        """
················································································

==Fix steady levels of variables==

················································································
        """
        pass

    @_pages.reference(category="definition", )
    def fix_change(
        self,
        names: Iterable[str] | str,
        /,
    ) -> None:
        """
················································································

==Fix steady changes of variables==

················································································
        """
        pass


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
    dates: Iterable[Period],
    base_span: tuple[Period],
    /,
) -> NoReturn | None:
    """
    """
    invalid = tuple(repr(d) for d in dates if d not in base_span)
    if invalid:
        raise _wrongdoings.IrisPieError(
            ("These date(s) are out of simulation span:", ) + invalid
        )


PlanSteady = SteadyPlan
Plan = SimulationPlan
PlanSimulate = SimulationPlan

