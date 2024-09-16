"""
Meta plans for dynamic simulations
"""


#[

from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Self, Any, Protocol, NoReturn, )
from types import (EllipsisType, )
import itertools as _it
import numpy as _np
import textwrap as _tw
import functools as _ft
import documark as _dm

from ..conveniences import copies as _copies
from ..dates import (Period, )
from ..series.main import (Series, )
from .. import wrongdoings as _wrongdoings
from . import _registers as _registers
from . import _pretty as _pretty
from . import _indexes as _indexes
from . import transforms as _transforms

#]


CHOOSE_TRANSFORM_CLASS = _transforms.CHOOSE_TRANSFORM_CLASS


__all__ = (
    "SimulationPlan", "PlanSimulate", "Plan",
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


#[
@_dm.reference(
    path=("structural_models", "simulation_plans.md", ),
    categories={
        "constructor": "Creating new simulation plans",
        "definition_simultaneous": "Defining exogenized and endogenized data points in [`Simultaneous` simulations](simultaneous.md#simulate)",
        "definition_sequential": "Defining exogenized and endogenized data points in [`Sequential` simulations](sequential.md#simulate)",
        "information": "Getting information about simulation plans",
        "information_simultaneous": "Getting information about simulation plans for [`Simultaneous` models](simultaneous.md)",
        "information_sequential": "Getting information about simulation plans for [`Sequential` models](sequential.md)",
    },
)
#]
class SimulationPlan(
    _registers.Mixin,
    _pretty.Mixin,
    _indexes.ItemMixin,
    _copies.Mixin,
):
    """
················································································

Simulation meta plans
======================

`SimulationPlan` objects are used to set up meta information about
conditioning assumptions for simulations of
[`Simultaneous`](simultaneous_modelsd) or [`Sequential`](sequential_models)
models. The simulation plans specify

* what variables to exogenize in what periods
* what shocks to endogenized in what periods (`Simultaneous` models only)
* what anticipation status to assign (`Simultaneous` models only)

The plans only contain meta information, not the actual data points for the
exogenized variables. The actual data points are expected to be included in
the input databox when the simulation is run.

················································································
    """
    #[

    _TABLE_FIELDS = ("NAME", "PERIOD(S)", "REGISTER", "TRANSFORM", "VALUE", )

    _registers = (
        "exogenized",
        "endogenized",
        "exogenized_anticipated",
        "exogenized_unanticipated",
        "endogenized_unanticipated",
        "endogenized_anticipated",
    )

    __slots__ = (
        ("base_span", )
        + tuple(f"_can_be_{r}" for r in _registers)
        + tuple(f"_{r}_register" for r in _registers)
        + tuple(f"default_{r}" for r in _registers)
    )

    @_dm.reference(
        category="constructor",
        call_name="SimulationPlan",
    )
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
        def default_value(*args, **kwargs, ):
            return [None] * self.num_periods
        self._initialize_registers(plannable, default_value, )

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
    @_dm.reference(category="property", )
    def start(self, /, ) -> Period:
        """==Start date of the simulation span=="""
        return self.base_span[0]

    start_period = start
    start_date = start

    @property
    @_dm.reference(category="property", )
    def end(self, /, ) -> Period:
        """==End date of the simulation span=="""
        return self.base_span[-1]

    end_period = end
    end_date = end

    @property
    @_dm.reference(category="property", )
    def num_periods(self, /, ) -> int:
        """==Number of periods in the simulation span=="""
        return len(self.base_span) if self.base_span is not None else 1

    @property
    @_dm.reference(category="property", )
    def frequency(self, /, ) -> str:
        """==Date frequency of the simulation span=="""
        return self.start.frequency

    @_dm.reference(category="definition_sequential", )
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
        r"""
................................................................................

==Exogenize certain LHS quantities at certain dates==

Exogenize certain LHS quantities at specified dates, setting them as
predetermined values within the simulation of
a [`Sequential` model](sequential.md). This method is used to control how
the model behaves during simulations by fixing certain variables to known
values.

    self.exogenize(
        dates,
        names,
        *,
        transform=None,
        when_data=False,
    )

### Input arguments ###

???+ input "self"
    The simulation plan in which data points will be exogenized.

???+ input "dates"
    A list of dates or `...` to apply to all dates at which the quantities 
    will be exogenized.

???+ input "names"
    A list of names or a single name, or `...` to apply to all names that 
    specifies which quantities to set as predetermined at the specified dates.

???+ input "transform"
    Specifies the transformation to apply to the exogenized quantities. If not
    specified, no transformation is applied. Available transformations include:

    * `None`: Exogenize the LHS variables as they are with no
    transformation.

    * `"log"`: Exogenize the natural logarithm of the LHS variables. Input
    time series needs to be prefixed with `log_`.

    * `"diff"`: Exogenize the first difference of the LHS variables. Input
    time series needs to be prefixed with `diff_`.

    * `"diff_log"`: Exogenize the first difference of the natural logarithm
    of the LHS variables. Input time series needs to be prefixed with
    `diff_log_`.

    * `"roc"`: The gross rate of change of the LHS variables from one
    period to the next. Input time series needs to be prefixed with `roc_`.

    * `"pct"`: The percentage change of the LHS variables from one period
    to the next. Input time series needs to be prefixed with `pct_`.

???+ input "when_data"
    Specifies whether the exogenization should only occur if a valid 
    value exists in the input data.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


................................................................................
        """
        transform = _transforms.resolve_transform(transform, **kwargs, )
        self._write_to_register("exogenized", dates, names, transform, )

    @_dm.reference(category="definition_simultaneous", )
    def exogenize_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        *,
        status: bool | int = True,
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
        self._write_to_register(
            "exogenized_anticipated",
            dates,
            names,
            status,
        )

    @_dm.reference(category="definition_simultaneous", )
    def exogenize_unanticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        *,
        status: bool | int = True,
    ) -> None:
        r"""
················································································

==Exogenize certain quantities at certain dates as unanticipated==

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
        self._write_to_register(
            "exogenized_unanticipated",
            dates,
            names,
            status,
        )

    @_dm.reference(category="information_sequential", )
    def get_exogenized_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names exogenized in a certain period==

················································································
        """
        return self._get_names_registered_in_period(
            self._exogenized_register,
            *args, **kwargs,
        )

    @_dm.reference(category="information_simultaneous", )
    def get_exogenized_unanticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names exogenized as unanticipated in a certain period==

················································································
        """
        return self._get_names_registered_in_period(
            self._exogenized_unanticipated_register,
            *args, **kwargs,
        )

    @_dm.reference(category="information_simultaneous", )
    def get_exogenized_anticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names exogenized as anticipated in a certain period==

················································································
        """
        return self._get_names_registered_in_period(
            self._exogenized_anticipated_register,
            *args, **kwargs,
        )

    # @_dm.reference(category="definition_sequential", )
    def endogenize(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        /,
    ) -> None:
        r"""
        """
        self._write_to_register("endogenized", dates, names, True, )

    @_dm.reference(category="definition_simultaneous", )
    def endogenize_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        *,
        status: bool | int = True,
    ) -> None:
        """
················································································

==Endogenize certain quantities at certain dates==

················································································
        """
        self._write_to_register(
            "endogenized_anticipated",
            dates,
            names,
            status,
        )

    @_dm.reference(category="definition_simultaneous", )
    def endogenize_unanticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        *,
        status: bool | int = True,
    ) -> None:
        """
················································································

==Endogenize certain quantities at certain dates==

················································································
        """
        self._write_to_register(
            "endogenized_unanticipated",
            dates,
            names,
            status,
        )

    @_dm.reference(category="information_simultaneous", )
    def get_endogenized_unanticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names endogenized as unanticipated in a certain period==

················································································
        """
        return self._get_names_registered_in_period(
            self._endogenized_unanticipated_register,
            *args,
            **kwargs,
        )

    @_dm.reference(category="information_simultaneous", )
    def get_endogenized_anticipated_in_period(self, *args, **kwargs, ) -> tuple[str, ...]:
        """
················································································

==Get names endogenized as anticipated in a certain period==

················································································
        """
        return self._get_names_registered_in_period(
            self._endogenized_unanticipated_register,
            *args,
            **kwargs,
        )

    @property
    def any_endogenized_unanticipated_except_start(self, /, ) -> bool:
        r"""
        True if there is any endogenized unanticipated point in the plan after the first period
        """
        return self._any_in_register_except_start("endogenized_unanticipated", )

    @property
    def any_endogenized_anticipated_except_start(self, /, ) -> bool:
        r"""
        True if there is any endogenized anticipated point in the plan after the first period
        """
        return self._any_in_register_except_start("endogenized_anticipated", )

    def _any_in_register_except_start(self, register_name, ) -> bool:
        r"""
        True if there is any unanticipated point in the plan in the second
        or later simulation period
        """
        return any(
            any(_is_active_status(i) for i in v[1:])
            for v in getattr(self, f"_{register_name}_register", ).values()
        )

    @property
    def is_empty(self, /, ) -> bool:
        """
        True if there are no exogenized or endogenized points in the plan
        """
        has_any_points = any(
            _has_points_in_register(self.get_register_by_name(n, ), )
            for n in self._registers
        )
        return not has_any_points

    # def swap(
    #     self,
    #     dates: Iterable[Period] | EllipsisType,
    #     pairs: Iterable[tuple[str, str]] | tuple[str, str],
    #     *args, **kwargs,
    # ) -> None:
    #     """
    #     """
    #     pairs = tuple(pairs)
    #     if not pairs:
    #         return
    #     if len(pairs) == 2 and isinstance(pairs[0], str) and isinstance(pairs[1], str):
    #         pairs = (pairs, )
    #     for pair in pairs:
    #         self.exogenize(dates, pair[0], *args, **kwargs, )
    #         self.endogenize(dates, pair[1], *args, **kwargs, )

    def swap_anticipated(
        self,
        dates: Iterable[Period] | EllipsisType,
        pairs: Iterable[tuple[str, str]] | tuple[str, str],
        *args, **kwargs,
    ) -> None:
        r"""
................................................................................

==Swap quantities as anticipated at certain dates==

Swap (exogenize and endogenize) quantities at certain dates. This method
exogenizes the first quantity in the pair and endogenizes the second
quantity in the pair at the specified dates. It is equivalent to calling
`exogenize_anticipated` and `endogenize_anticipated` separately.

    self.swap_anticipated(
        dates,
        pairs,
    )


### Input arguments ###

???+ input "self"
    The simulation plan in which data points will be exogenized and
    endogenized.

???+ input "dates"
    Dates at which the quantities will be exogenized and endogenized.

???+ input "pairs"
    A list of pairs of names to exogenize and endogenize at the specified
    dates.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


................................................................................
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
................................................................................

==Swap quantities as unanticipated at certain dates==

Swap (exogenize and endogenize) quantities at certain dates. This method
exogenizes the first quantity in the pair and endogenizes the second
quantity in the pair at the specified dates. It is equivalent to calling
`exogenize_unanticipated` and `endogenize_unanticipated` separately.

    self.swap_unanticipated(
        dates,
        pairs,
    )


### Input arguments ###

???+ input "self"
    The simulation plan in which data points will be exogenized and
    endogenized.

???+ input "dates"
    Dates at which the quantities will be exogenized and endogenized.

???+ input "pairs"
    A list of pairs of names to exogenize and endogenize at the specified
    dates.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


................................................................................
        """
        pairs = tuple(pairs)
        if not pairs:
            return
        if len(pairs) == 2 and isinstance(pairs[0], str) and isinstance(pairs[1], str):
            pairs = (pairs, )
        for pair in pairs:
            self.exogenize_unanticipated(dates, pair[0], *args, **kwargs, )
            self.endogenize_unanticipated(dates, pair[1], *args, **kwargs, )

    def get_register_as_bool_array(
        self: Self,
        register_name: str,
        names: str | Iterable[str] | EllipsisType = ...,
        periods: Iterable[Period] | EllipsisType = ...,
    ) -> _np.ndarray:
        """
        """
        register = self.get_register_by_name(register_name, )
        per_indexes = self._get_per_indexes(periods, )
        names = self._resolve_register_names(register, names, )
        num_names, num_pers = len(names), len(per_indexes)
        #
        if not names or not per_indexes:
            return _np.zeros((num_names, num_pers, ), dtype=bool, )
        #
        def get_points_for_name(name: str, ) -> tuple[bool, ...]:
            return tuple(
                register[name][t] if t is not None else False
                for t in per_indexes
            )
        return _np.array(tuple(
            get_points_for_name(n, )
            for n in names
        ), dtype=bool, )

    def get_registers_as_bool_arrays(
        self,
        periods: tuple[Period, ...] | EllipsisType = ...,
        register_names: Iterable[str] | EllipsisType = ...,
    ) -> dict[str, _np.ndarray]:
        """
        """
        #[
        if periods is Ellipsis:
            periods = tuple(self.base_span)
        #
        if register_names is Ellipsis:
            register_names = tuple(self._registers)
        #
        get_register_as_bool_array = _ft.partial(
            self.get_register_as_bool_array,
            names=...,
            periods=periods,
        )
        return {
            n: get_register_as_bool_array(register_name=n, )
            for n in register_names
        }
        #]

    def _get_per_indexes(
        self,
        periods: Iterable[Period] | EllipsisType,
        /,
    ) -> tuple[int | None, ...]:
        """
        """
        if periods is ...:
            return tuple(range(self.num_periods, ), )
        else:
            return tuple(
                t - self.start if self._is_per_in_span(t, ) else None
                for t in periods
            )

    def _is_per_in_span(self, per: Period, ) -> bool:
        return per >= self.start and per <= self.end

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

    def _write_to_register(
        self,
        register_name: str,
        periods: Iterable[Period] | EllipsisType,
        names: Iterable[str] | str | EllipsisType,
        new_status: Any,
    ) -> None:
        """
        """
        register = self.get_register_by_name(register_name, )
        names = self._resolve_register_names(register, names, )
        per_indexes, *_ = self._get_period_indexes(periods, )
        for n in names:
            for t in per_indexes:
                register[n][t] = new_status

    def _get_period_indexes(
        self,
        periods: Iterable[Period] | EllipsisType,
        /,
    ) -> tuple[tuple[int, ...], tuple[Period, ...]]:
        """
        """
        if periods is Ellipsis:
            periods = tuple(self.base_span)
        if hasattr(periods, "resolve"):
            periods = periods.resolve(self, )
        catch_invalid_periods(periods, self.base_span, )
        period_indexes = tuple(d - self.start for d in periods)
        return period_indexes, periods

    def _add_register_to_table(
        self,
        table,
        register: dict,
        action: str,
        db: Databox | None = None,
        **kwargs,
    ) -> None:
        """
        """
        def _get_status_symbol(status, ):
            return status.symbol if hasattr(status, "symbol") else _PRETTY_SYMBOL.get(status, "")
        #
        def _get_value(db, name, date, status, ):
            missing_str = Series._missing_str
            databox_name = (
                status.resolve_databox_name(name, )
                if hasattr(status, "resolve_databox_name")
                else name
            )
            try:
                value = db[databox_name][date][0, 0]
            except:
                return missing_str
            if _np.isnan(value):
                return missing_str
            return f"{value:g}"
        #
        all_rows = (
            (k, str(date), action, _get_status_symbol(status), _get_value(db, k, date, status, ), )
            for k, v in register.items()
            for status, date in zip(v, self.base_span)
            if status is not None and status is not False
        )
        #
        all_rows = sorted(all_rows, key=lambda row: (row[0], row[1], ), )
        row_groups = _it.groupby(all_rows, key=lambda row: (row[0], row[2], row[3], ), )
        for _, g in row_groups:
            representative = _create_representative_for_table_rows(tuple(g), )
            table.add_row(representative, )

    for n in _registers:
        exec(_tw.dedent(f"""
            def get_{n}(self, ) -> tuple[Period, ...]:
                return _get_registered_periods(self._{n}_register, self.base_span, )
        """))

    #]


def _get_registered_periods(
    register: dict[str, Any],
    base_span: tuple[Period],
    /,
) -> dict[str, tuple[Period]]:
    """
    """
    return {
        k: tuple(
            period
            for period, status in zip(base_span, v)
            if status is not None and status is not False
        )
        for k, v in register.items()
    }


def catch_invalid_periods(
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


def _is_active_status(value: Any, /, ) -> bool:
    return (value is not None) and (value is not False)


def _has_points_in_register(register: dict, /, ) -> bool:
    return any(
        any(_is_active_status(i) for i in v)
        for v in register.values()
    )


def _create_representative_for_table_rows(rows, ):
    if len(rows) == 1:
        return rows[0]
    else:
        return (rows[0][0], rows[0][1] + ">>" + rows[-1][1], *rows[0][2:], )


Plan = SimulationPlan
PlanSimulate = SimulationPlan


_PRETTY_SYMBOL = {
    None: "",
    True: "⋅",
    False: "",
}

