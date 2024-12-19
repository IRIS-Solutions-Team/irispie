"""
Frequency conversion of time series
"""


#[

from __future__ import annotations

from typing import (Self, Callable, )
from types import (EllipsisType, )
import numpy as _np
import functools as _ft
import operator as _op
import statistics as _st
import documark as _dm

from .. import dates as _dates
from . import arip as _arip
from . import _functionalize

#]


__all__ = (
    "convert_roc",
    "convert_pct",
)


_builtin_min = min
_builtin_max = max
_builtin_sum = sum


_DEFAULT_METHOD = "mean"


class Inlay:
    r"""
    ................................................................................
    ==Inlay Class==

    Provides methods for aggregating and disaggregating time series data 
    between different frequencies. Supports multiple aggregation and 
    interpolation methods, allowing flexible transformations of time series data.

    ................................................................................
    """
    #[

    @_dm.reference(category="conversion", )
    def aggregate(
        self,
        target_freq: _dates.Frequency,
        /,
        method: Literal["mean", "sum", "first", "last", "min", "max"] | Callable | None = None,
        remove_missing: bool = False,
        select: list[int] | None = None,
    ) -> None:
        """
················································································

==Aggregate time series to a lower frequency==


### Function form for creating new time `Series` objects ###

    new = irispie.aggregate(
        self,
        target_freq,
        /,
        method="mean",
        remove_missing=False,
        select=None,
    )


### Class method form for creating new time `Series` objects ###

    self.aggregate(
        target_freq,
        /,
        method="mean",
        remove_missing=False,
        select=None,
    )


### Input arguments ###


???+ input "target_freq"
    The new frequency to which the original time series will be diaggregated.

???+ input "method"
    Aggregation method, i.e. a function applied to the high-frequency
    values within each low-frequency period:

    | Method    | Description
    |-----------|-------------
    | "mean"    | Arithmetic average of high-frequency values
    | "sum"     | Sum of high-frequency values
    | "prod"    | Product of high-frequency values
    | "first"   | Value in the first high-frequency period
    | "last"    | Value in the last high-frequency period
    | "min"     | Minimum of high-frequency values
    | "max"     | Maximum of high-frequency values

???+ input "remove_missing"
    Remove missing values from the high-frequency data before
    applying the aggregation `method`.

???+ input "select"
    Select only the high-frequency values at the specified indexes;
    `select=None` means all values are used.


### Returns ###

???+ returns "self"
    The original time `Series` object with the aggregated data.

???+ returns "new"
    A new time `Series` object with the aggregated data.

················································································
        """
        method = method or _DEFAULT_METHOD
        method_func = (
            _AGGREGATION_METHOD_RESOLUTION[method]
            if isinstance(method, str) else method
        )
        #
        if target_freq == self.frequency:
            return
        if target_freq > self.frequency or target_freq is _dates.Frequency.UNKNOWN or self.frequency is _dates.Frequency.UNKNOWN:
            raise ValueError(f"Cannot aggregate from {self.frequency} frequency to {target_freq} frequency")
        #
        new_dater_class = _dates.PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[target_freq]
        #
        aggregate_within_data_func = _ft.partial(
            _aggregate_within_data,
            select,
            remove_missing,
            method_func,
        )
        #
        if self.frequency.is_regular:
            aggregate_func = _aggregate_regular_to_regular
        elif self.frequency is _dates.Frequency.DAILY:
            aggregate_func = _aggregate_daily_to_regular
        #
        new_start_date, new_data = aggregate_func(self, new_dater_class, aggregate_within_data_func, )
        self._replace_start_and_values(new_start_date, new_data, )

    @_dm.reference(category="conversion", )
    def disaggregate(
        self,
        target_freq: _dates.Frequency,
        /,
        method: str = "flat",
        **kwargs,
    ) -> Self:
        r"""
················································································

==Disaggregate time series to a higher frequency==


### Function form for creating new time `Series` objects ###

    new = irispie.disaggregate(
        self,
        target_freq,
        /,
        method="flat",
    )


### Class method form for changing existing time `Series` objects in-place ###

    self.disaggregate(
        target_freq,
        /,
        method="flat",
        model=None,
    )


### Input arguments ###


???+ input "target_freq"
    The new frequency to which the original time series will be aggregated.

???+ input "method"
    Aggregation method, i.e. a function applied to the high-frequency
    values within each low-frequency period:

    | Method    | Description
    |-----------|-------------
    | "flat"    | Repeat the high-frequency values
    | "first"   | Place the low-frequency value in the first high-frequency period
    | "last"    | Place the low-frequency value in the last high-frequency period
    | "arip"    | Interpolate using a smooth autoregressive process


### Returns ###


???+ returns "new"
    A new time `Series` object with the disaggregated data.



### Details ###

???+ details "ARIP algorithm"

    The `method="arip" setting invokes an interpolation method that assumes the
    underlying high-frequency process to be an autoregression. The method can be
    described in its state-space recursive form, although the numerical
    implementation is stacked-time.

    The `"rate"` model:

    $$
    \begin{gathered}
    x_t = \rho \, x_{t-1} + \epsilon_t \\[10pt]
    y_t = Z \, x_t \\[10pt]
    \epsilon_t \sim N(0, \sigma_t^2)
    \end{gathered}
    $$

    The `"diff"` model:

    $$
    \begin{gathered}
    x_t = x_{t-1} + c + \epsilon_t \\[10pt]
    y_t = Z \, x_t \\[10pt]
    \epsilon_t \sim N(0, 1)
    \end{gathered}
    $$

    where

    * $x_t$ is the underlying high-frequency process;

    * $y_t$ is the observed low-frequency time series;

    * $Z$ is an aggregation vector depending on the `aggregation` specification,

    | Aggregation | $Z$ vector
    |-------------|-----------
    | "sum"       | $(1, 1, \ldots, 1)$
    | "mean"      | $\tfrac{1}{n}\,(1, 1, \ldots, 1)$
    | "first"     | $(1, 0, \ldots, 0)$
    | "last"      | $(0, 0, \ldots, 1)$

    * $\rho$ is a gross rate of change estimated as the average rate of change
    in the observed series, $y_t$, and converted to high frequency;

    * $c$ is a constant estimated as the average difference in the observed
    series, $y_t$, and converted to high frequency;

    * $\sigma_t$ is a time-varying standard deviation of the high-frequency process, set to $\sigma_0 = 1$, and $\sigma_t = \rho \, \sigma_{t-1}$.

················································································
        """
        method_func = _CHOOSE_DISAGGREGATION_METHOD[method]
        #
        if target_freq == self.frequency:
            return
        #
        if target_freq < self.frequency \
            or target_freq is _dates.Frequency.UNKNOWN \
            or self.frequency is _dates.Frequency.UNKNOWN:
            #
            raise _wrongdoings.IrisPieCritical(
                f"Cannot disaggregate from {self.frequency} frequency to {target_freq} frequency"
            )
        #
        new_dater_class = _dates.PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[target_freq]
        new_start_date, new_data, *_ = method_func(self, new_dater_class, **kwargs, )
        self._replace_start_and_values(new_start_date, new_data, )

    #]


for n in ("aggregate", "disaggregate", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _aggregate_daily_to_regular(
    self,
    new_dater_class: type,
    aggregate_within_data_func: Callable,
) -> Self:
    r"""
    ................................................................................
    ==Aggregate Daily Time Series to Regular Frequency==

    Aggregates data from daily frequency to a lower regular frequency (e.g., monthly, 
    yearly).

    This function handles the specific requirements for processing daily time series.

    ................................................................................

    ### Input arguments ###
    ???+ input "self"
        The current time series object.

    ???+ input "new_dater_class"
        The date class corresponding to the target frequency.

    ???+ input "aggregate_within_data_func"
        A callable function for aggregating data within each low-frequency period.

    ### Returns ###
    ???+ returns "tuple[Dater, numpy.ndarray]"
        A tuple containing the new start date and the aggregated data array.

    ### Example ###
    ```python
        new_start_date, aggregated_data = _aggregate_daily_to_regular(
            series, new_dater_class, aggregate_func
        )
    ```
    ................................................................................
    """
    #[
    start_date = self.start_date.create_soy()
    end_date = self.end_date.create_eoy()
    from_until = (start_date, end_date, )
    start_year = self.start_date.get_year()
    end_year = self.end_date.get_year()
    new_start_date = new_dater_class.from_year_period(start_year, 1)
    new_end_date = new_dater_class.from_year_period(end_year, "end")
    get_slice_func = lambda t: slice(
        t.to_daily(position="start", ) - start_date,
        t.to_daily(position="end", ) - start_date + 1,
    )
    new_data = tuple(
        tuple(
            aggregate_within_data_func(data_variant[get_slice_func(t)])
            for data_variant in self.iter_own_data_variants_from_until(from_until, )
        )
        for t in _dates.Ranger(new_start_date, new_end_date)
    )
    #
    new_data = _np.array(new_data, dtype=self.data_type, )
    return new_start_date, new_data
    #]


def _aggregate_regular_to_regular(
    self,
    new_dater_class: type,
    aggregate_within_data_func: Callable,
    /,
) -> tuple[Dater, _np.ndarray]:
    r"""
    ................................................................................
    ==Aggregate Regular Time Series==

    Aggregates data from one regular frequency to a lower regular frequency.

    This function is used internally by the `aggregate` method to process time series 
    with regular frequencies.

    ................................................................................

    ### Input arguments ###
    ???+ input "self"
        The current time series object.

    ???+ input "new_dater_class"
        The date class corresponding to the target frequency.

    ???+ input "aggregate_within_data_func"
        A callable function for aggregating data within each low-frequency period.

    ### Returns ###
    ???+ returns "tuple[Dater, numpy.ndarray]"
        A tuple containing the new start date and the aggregated data array.

    ### Example ###
    ```python
        new_start_date, aggregated_data = _aggregate_regular_to_regular(
            series, new_dater_class, aggregate_func
        )
    ```
    ................................................................................
    """
    #[
    start_year = self.start_date.get_year()
    start_date = self.start_date.create_soy()
    end_date = self.end_date.create_eoy()
    new_start_date = new_dater_class.from_year_period(start_year, 1)
    target_freq = new_dater_class.frequency
    self_data = self.get_data_from_until((start_date, end_date))
    factor = self.frequency.value // target_freq
    #
    # Loop over variants, create within-period data, and aggregate
    multi_output_data_transposed = []
    for data_variant in self_data.T:
        output_data_transposed = tuple(
            aggregate_within_data_func(within_data)
            for within_data in data_variant.reshape((-1, factor), )
        )
        multi_output_data_transposed.append(output_data_transposed, )
    #
    multi_output_data = _np.array(multi_output_data_transposed, dtype=self.data_type, ).T
    return new_start_date, multi_output_data
    #]


def _disaggregate_flat(
    self,
    high_dater_class: type,
    /,
) -> tuple[Dater, _np.ndarray]:
    r"""
    ................................................................................
    ==Flat Disaggregation==

    Disaggregates low-frequency data into higher-frequency periods by repeating 
    values for all high-frequency periods.

    ................................................................................

    ### Input arguments ###
    ???+ input "self"
        The current time series object.

    ???+ input "high_dater_class"
        The date class corresponding to the target high frequency.

    ### Returns ###
    ???+ returns "tuple[Dater, numpy.ndarray]"
        A tuple containing the new start date and the disaggregated data array.

    ### Example ###
    ```python
        new_start_date, disaggregated_data = _disaggregate_flat(
            series, high_dater_class
        )
    ```
    ................................................................................
    """
    #[
    high_freq = high_dater_class.frequency
    high_start_date = high_dater_class.from_ymd(*self.start_date.to_ymd(position="start", ), )
    factor = high_freq.value // self.frequency.value
    high_data = _np.repeat(self.data, factor, axis=0, )
    return high_start_date, high_data, factor
    #]


def _disaggregate_first(
    self,
    high_dater_class: type,
    /,
) -> tuple[Dater, _np.ndarray]:
    r"""
    ................................................................................
    ==First Value Disaggregation==

    Disaggregates low-frequency data into higher-frequency periods by placing the 
    value in the first high-frequency period of each interval, setting the remaining 
    periods to `NaN`.

    ................................................................................

    ### Input arguments ###
    ???+ input "self"
        The current time series object.

    ???+ input "high_dater_class"
        The date class corresponding to the target high frequency.

    ### Returns ###
    ???+ returns "tuple[Dater, numpy.ndarray]"
        A tuple containing the new start date and the disaggregated data array.

    ### Example ###
    ```python
        new_start_date, disaggregated_data = _disaggregate_first(
            series, high_dater_class
        )
    ```
    ................................................................................
    """
    #[
    high_start_date, high_data, factor = _disaggregate_flat(self, high_dater_class, )
    for i in range(1, factor, ):
        high_data[i::factor, :] = _np.nan
    return high_start_date, high_data
    #]


def _disaggregate_last(
    self,
    high_dater_class: type,
    /,
) -> tuple[Dater, _np.ndarray]:
    r"""
    ................................................................................
    ==Last Value Disaggregation==

    Disaggregates low-frequency data into higher-frequency periods by placing the 
    value in the last high-frequency period of each interval, setting the remaining 
    periods to `NaN`.

    ................................................................................

    ### Input arguments ###
    ???+ input "self"
        The current time series object.

    ???+ input "high_dater_class"
        The date class corresponding to the target high frequency.

    ### Returns ###
    ???+ returns "tuple[Dater, numpy.ndarray]"
        A tuple containing the new start date and the disaggregated data array.

    ### Example ###
    ```python
        new_start_date, disaggregated_data = _disaggregate_last(
            series, high_dater_class
        )
    ```
    ................................................................................
    """
    #[
    high_start_date, high_data, factor = _disaggregate_flat(self, high_dater_class, )
    for i in range(0, factor-1, ):
        high_data[i::factor, :] = _np.nan
    return high_start_date, high_data, factor
    #]


def _aggregate_within_data(
    select: list[int] | None,
    remove_missing: bool,
    method_func: Callable,
    within_data: _np.ndarray,
    /,
) -> _np.ndarray:
    r"""
    ................................................................................
    ==Aggregate Data Within a Period==

    Processes data for a single period, applying the specified aggregation method 
    and handling missing values or specific selections.

    This function is called during aggregation operations to compute aggregated values 
    for each target frequency period.

    ................................................................................

    ### Input arguments ###
    ???+ input "select"
        An optional list of indexes to select specific high-frequency values.

    ???+ input "remove_missing"
        A boolean flag. If `True`, removes missing values before applying the 
        aggregation method.

    ???+ input "method_func"
        The aggregation function to apply to the data (e.g., mean, sum).

    ???+ input "within_data"
        A numpy array containing the data for a single high-frequency period.

    ### Returns ###
    ???+ returns "numpy.ndarray"
        The aggregated value for the period.

    ### Example ###
    ```python
        aggregated_value = _aggregate_within_data(
            select=None, 
            remove_missing=True, 
            method_func=_np.mean, 
            within_data=data
        )
    ```
    ................................................................................
    """
    #[
    if select is not None:
        select = tuple(select)
        within_data = within_data[select]
    if remove_missing:
        within_data = within_data[~_np.isnan(within_data)]
    return method_func(within_data) if within_data.size > 0 else _np.nan
    #]


_AGGREGATION_METHOD_RESOLUTION = {
    "mean": _st.mean,
    "geometric_mean": _st.geometric_mean,
    "sum": _builtin_sum,
    "prod": _np.prod,
    "first": _op.itemgetter(0),
    "last": _op.itemgetter(-1),
    "min": _builtin_min,
    "max": _builtin_max,
}


_CHOOSE_DISAGGREGATION_METHOD = {
    "flat": _disaggregate_flat,
    "first": _disaggregate_first,
    "last": _disaggregate_last,
    "arip": _arip.disaggregate_arip,
}


def convert_roc(
    roc: Real,
    from_freq: _dates.Frequency,
    to_freq: _dates.Frequency,
    /,
) -> Real:
    r"""
    ................................................................................
    ==Convert Rate of Change Between Frequencies==

    Converts a rate of change (`roc`) value from one frequency to another using 
    exponential scaling.

    ................................................................................

    ### Input arguments ###
    ???+ input "roc"
        The rate of change value to be converted.

    ???+ input "from_freq"
        The frequency of the input rate of change.

    ???+ input "to_freq"
        The target frequency to which the rate of change should be converted.

    ### Returns ###
    ???+ returns "Real"
        The rate of change value at the target frequency.

    ### Example ###
    ```python
        converted_roc = convert_roc(1.05, _dates.Frequency.MONTHLY, _dates.Frequency.YEARLY)
        print(converted_roc)
    ```
    ................................................................................
    """
    return float(roc) ** (float(from_freq) / float(to_freq))


def convert_pct(
    pct: Real,
    from_freq: _dates.Frequency,
    to_freq: _dates.Frequency,
    /,
) -> Real:
    r"""
    ................................................................................
    ==Convert Percentage Change Between Frequencies==

    Converts a percentage change (`pct`) value from one frequency to another using 
    exponential scaling.

    ................................................................................

    ### Input arguments ###
    ???+ input "pct"
        The percentage change value to be converted.

    ???+ input "from_freq"
        The frequency of the input percentage change.

    ???+ input "to_freq"
        The target frequency to which the percentage change should be converted.

    ### Returns ###
    ???+ returns "Real"
        The percentage change value at the target frequency.

    ### Example ###
    ```python
        converted_pct = convert_pct(5.0, _dates.Frequency.MONTHLY, _dates.Frequency.YEARLY)
        print(converted_pct)
    ```
    ................................................................................
    """
    return 100*(convert_roc(1 + float(pct)/100, from_freq, to_freq) - 1)


def convert_diff(
    diff: Real,
    from_freq: _dates.Frequency,
    to_freq: _dates.Frequency,
    /,
) -> Real:
    r"""
    ................................................................................
    ==Convert Differences Between Frequencies==

    Converts a difference (`diff`) value from one frequency to another using scaling 
    proportional to the ratio of frequencies.

    ................................................................................

    ### Input arguments ###
    ???+ input "diff"
        The difference value to be converted.

    ???+ input "from_freq"
        The frequency of the input difference.

    ???+ input "to_freq"
        The target frequency to which the difference should be converted.

    ### Returns ###
    ???+ returns "Real"
        The difference value at the target frequency.

    ### Example ###
    ```python
        converted_diff = convert_diff(10, _dates.Frequency.MONTHLY, _dates.Frequency.YEARLY)
        print(converted_diff)
    ```
    ................................................................................
    """
    return float(diff) * (float(from_freq) / float(to_freq))

