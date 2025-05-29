"""
Temporal change functions
"""


#[
from __future__ import annotations

from typing import (Self, Callable, )
from numbers import (Real, )
import numpy as _np
import documark as _dm

from ..dates import (Span, )
from ._functionalize import FUNC_STRING
from ._categories import CATEGORIES
#]


__all__ = []


class Inlay:
    """
    """
    #[

    @_dm.reference(
        category=None,
        call_name=CATEGORIES["temporal_change"],
        call_name_is_code=False,
        priority=20,
    )
    def temporal_change(
        self,
        by: int | str,
        func: Callable,
        /,
        **kwargs,
    ) -> None:
        r"""
................................................................................

Overview of temporal change functions with flexible time shifts:


| Function      | Description
|---------------|-------------
| `diff`        | First difference, $y_t = x_t - x_s$ 
| `diff_log`    | First difference of logs, $y_t = \log x_t - \log x_s$ 
| `pct`         | Percent change, $y_t = 100 \cdot (x_t/x_s - 1)$ 
| `roc`         | Gross rate of change, $y_t = x_t/x_s$ 


Overview of temporal change functions with fixed time shifts:

| Function      | Description
|---------------|-------------
| `adiff`       | Annualized first difference, $y_t = a \cdot (x_t - x_{t-1})$ 
| `adiff_log`   | Annualized first difference of logs, $y_t = a \cdot (\log x_t - \log x_{t-1})$ 
| `apct`        | Annualized percent change, $y_t = 100 \cdot \left[ (x_t/x_{t-1})^a - 1 \right]$ 
| `aroc`        | Annualized gross rate of change, $y_t = (x_t/x_{t-1})^a$ 

where

* $x_t$ is the value of the time series at $t$;

* $x_s$ is the reference value of the time series in some preceding period $s$;
the value of $s$ depends on the (optional) input argument `shift` (whose default
value is `shift=-1`); see the explanation below;

* $a$ is an annualization factor, determined by the frequency of the time
series (i.e. the number of segments of a give frequency within a year: 1 for
yearly frequency, 2 for semi-annual frequency, 4 for quarterly frequency, 12 for
monthly frequency, 365 for daily frequency).

The `shift` input argument can be either a negative integer or a string to cover
some specific temporal change calculations:

* If `shift` is a negative integer, the reference period is $s := t-k$ where $k$ is
the negative value of `shift`; positive values (leads) of `shift` are not allowed.

* `shift="yoy"` means a year-on-year change, whereby the time shift is set to
the negative of the frequency of the time series, $s := t-a$ with $a$ being
defined above (the annualization factor).

* `shift="soy"` means a change over the start of the current year; in this case,
the reference period $s$ is set to the first segment of the current year (i.e.
the 1st half-year, the 1st quarter, the 1st month, etc.)

* `shift="eopy"` means a change over the end of the previous year; in this case,
the reference period $s$ is set to the last segment of the previous year (i.e.
the 2nd half-year, the 4th quarter, the 12th month, etc.).

* `shift="tty"` means a change throughout the year; this is equivalent to
`shift=-1` except for any start-of-year periods; in start-of-year periods, the
value of the resulting series is unchanged; for instance, `diff` applied to a
quarterly series with `shift="tty"` will return a series in which the Q1 value
will be unchanged, the Q2 value will be the difference between Q2 and Q1, the Q3
value will be the difference between Q3 and Q2, and the Q4 value will be the
difference between Q4 and Q3.


### Functional forms creating a new Series object ###

```
new = irispie.diff(self, shift=-1)
new = irispie.diff_log(self, shift=-1)
new = irispie.pct(self, shift=-1)
new = irispie.roc(self, shift=-1)
new = irispie.adiff(self)
new = irispie.adiff_log(self)
new = irispie.apct(self)
new = irispie.aroc(self)
```


### Class methods changing an existing Series object in-place ###

```
self.diff(shift=-1)
self.difflog(shift=-1)
self.pct(shift=-1)
self.roc(shift=-1)
self.adiff()
self.adifflog()
self.apct()
self.aroc()
```



### Input arguments ###

???+ input "self"
    Time series on whose data a temporal change function is calculated (see
    the overview table above).

???+ input "shift"
    A negative integer or a string determining a time lag at which the temporal change
    function is calculated. If `shift=None` (or not specified), `shift` is
    set to `-1`. If `shift` is a string, it must be one of the following:

    * `"yoy"`: year-on-year change
    * `"soy"`: change over the start of the current year
    * `"eopy"`: change over the end of the previous year
    * `"tty"`: change throughout the year


### Returns ###

???+ returns "new"
    New time series with data calculated as a temporal change function of
    the original data.

???+ returns "self"
    Time series with data replaced by a temporal change function of the
    original data.

................................................................................
        """
        _catch_invalid_shift(by, )
        other = self.copy()
        other.shift(by, **kwargs, )
        self._binop(other, func, new=self, )

    @_dm.reference(
        category=None,
        call_name=CATEGORIES["temporal_change_conversion"],
        call_name_is_code=False,
        priority=19,
    )
    def temporal_change_conversion(self, ):
        r"""
................................................................................

Overview of temporal change conversions:

| Function          | Description
|-------------------|-------------
| `roc_from_pct`    | Convert percent change to gross rate of change
| `pct_from_roc`    | Convert gross rate of change to percent change
| `pct_from_apct`   | Convert annualized percent change to percent change
| `roc_from_apct`   | Convert annualized percent change to gross rate of change
| `roc_from_aroc`   | Convert annualized gross rate of change to gross rate of change

### Functional forms changing an existing Series object in-place ###

```
new = irispie.roc_from_pct(self)
new = irispie.pct_from_roc(self)
new = irispie.pct_from_apct(self)
new = irispie.roc_from_apct(self)
new = irispie.roc_from_aroc(self)
```

### Class methods changing an existing Series object in-place ###

```
self.roc_from_pct()
self.pct_from_roc()
self.pct_from_apct()
self.roc_from_apct()
self.roc_from_aroc()
```

................................................................................
        """
        pass

    @_dm.reference(category="temporal_change", )
    def diff(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        r"""
................................................................................

==First difference==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        self.temporal_change(shift, lambda x, y: x - y, neutral_value=0, )

    @_dm.reference(category="temporal_change", )
    def adiff(self, ) -> None:
        r"""
................................................................................

==Annualized first difference==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        shift = -1
        factor = self.frequency.value or 1
        self.temporal_change(shift, lambda x, y: factor*(x - y), neutral_value=0, )

    @_dm.reference(category="temporal_change", )
    def diff_log(
        self,
        shift: int | str = -1,
    ) -> None:
        r"""
................................................................................

==First difference of logs==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        self.temporal_change(shift, lambda x, y: _np.log(x) - _np.log(y), neutral_value=0, )

    @_dm.reference(category="temporal_change", )
    def adiff_log(self, ) -> None:
        r"""
................................................................................

==Annualized first difference of logs==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        shift = -1
        factor = self.frequency.value or 1
        self.temporal_change(shift, lambda x, y: factor*(_np.log(x) - _np.log(y)), neutral_value=0, )

    @_dm.reference(category="temporal_change", )
    def roc(
        self,
        shift: int | str = -1,
    ) -> None:
        r"""
................................................................................

==Gross rate of change==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        self.temporal_change(shift, lambda x, y: x/y, neutral_value=1, )

    @_dm.reference(category="temporal_change", )
    def aroc(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Annualized gross rate of change==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        shift = -1
        factor = self.frequency.value or 1
        self.temporal_change(shift, lambda x, y: (x/y)**factor, neutral_value=1, )

    @_dm.reference(category="temporal_change", )
    def pct(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        r"""
................................................................................

==Percent change==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        self.temporal_change(shift, lambda x, y: 100*(x/y - 1), neutral_value=None, )

    @_dm.reference(category="temporal_change", )
    def apct(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Annualized percent change==

See documentation for [temporal change calculations](#temporal-change-calculations) or
`help(irispie.Series.temporal_change)`.

................................................................................
        """
        shift = -1
        factor = self.frequency.value or 1
        self.temporal_change(shift, lambda x, y: 100*((x/y)**factor - 1), neutral_value=None, )

    @_dm.reference(category="temporal_change_conversion", )
    def roc_from_pct(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Gross rate of change from percent change==

See documentation for [converting measures of temporal
change](#temporal-change-conversion).

................................................................................
        """
        self.data = 1 + self.data/100

    @_dm.reference(category="temporal_change_conversion", )
    def pct_from_roc(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Percent change from gross rate of change==

See documentation for [converting measures of temporal
change](#temporal-change-conversion).

................................................................................
        """
        self.data = 100*(self.data - 1)

    @_dm.reference(category="temporal_change_conversion", )
    def pct_from_apct(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Percent change from annualized percent change==

See documentation for [converting measures of temporal
change](#temporal-change-conversion).

................................................................................
        """
        factor = self.frequency.value or 1
        self.data = 100*((1 + self.data/100)**(1/factor) - 1)

    @_dm.reference(category="temporal_change_conversion", )
    def roc_from_apct(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Gross rate of change from annualized percent change==

See documentation for [converting measures of temporal
change](#temporal-change-conversion).

................................................................................
        """
        factor = self.frequency.value or 1
        self.data = (1 + self.data/100)**(1/factor)

    @_dm.reference(category="temporal_change_conversion", )
    def roc_from_aroc(
        self,
        /,
    ) -> None:
        r"""
................................................................................

==Gross rate of change from annualized gross rate of change==

See documentation for [converting measures of temporal
change](#temporal-change-conversion).

................................................................................
        """
        factor = self.frequency.value or 1
        self.data = self.data**factor

    @_dm.reference(category="temporal_cumulation", )
    def cum_diff(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Cumulation of first differences==

See documentation for [temporal cumulation calculations](#temporal-cumulation-calculations) or
`help(irispie.Series.temporal_cumulation)`.

................................................................................
        """
        self.temporal_cumulation("diff", *args, **kwargs, )

    @_dm.reference(category="temporal_cumulation", )
    def cum_diff_log(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Cumulation of first differences of logs==

See documentation for [temporal cumulation calculations](#temporal-cumulation-calculations) or
`help(irispie.Series.temporal_cumulation)`.

................................................................................
        """
        self.temporal_cumulation("diff_log", *args, **kwargs, )

    @_dm.reference(category="temporal_cumulation", )
    def cum_pct(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Cumulation of percent changes==

See documentation for [temporal cumulation calculations](#temporal-cumulation-calculations) or
`help(irispie.Series.temporal_cumulation)`.

................................................................................
        """
        self.temporal_cumulation("pct", *args, **kwargs, )

    @_dm.reference(category="temporal_cumulation", )
    def cum_roc(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Cumulation of gross rates of change==

See documentation for [temporal cumulation calculations](#temporal-cumulation-calculations) or
`help(irispie.Series.temporal_cumulation)`.

................................................................................
        """
        self.temporal_cumulation("roc", *args, **kwargs, )

    @_dm.reference(
        category=None,
        call_name=CATEGORIES["temporal_cumulation"],
        call_name_is_code=False,
        priority=20,
    )
    def temporal_cumulation(
        self,
        func_name: str,
        shift: int | str = -1,
        initial: Real | Self | None = None,
        span: Span | None = None,
    ) -> None:
        r"""
................................................................................

Overview of temporal cumulation calculations:

| Function      | Description
|---------------|-------------
| `cum_diff`    | Cumulative difference, $y_t = \sum_{i=0}^{t} (x_i - x_{i-k})$
| `cum_diff_log`| Cumulative difference of logs, $y_t = \sum_{i=0}^{t} (\log x_i - \log x_{i-k})$
| `cum_pct`     | Cumulative percent change, $y_t = 100 \sum_{i=0}^{t} (x_i/x_{i-k} - 1)$
| `cum_roc`     | Cumulative gross rate of change, $y_t = \prod_{i=0}^{t} (x_i/x_{i-k})$

where

* $k$ is a time shift (time lag) with which the temporal change is
calculated, determined by the negative value of the `shift` input argument.

* $t$ is the end date of the time series.


### Functional forms creating new Series objects ###

```
new = irispie.cum_diff(self, shift=-1, initial=None, span=None)
new = irispie.cum_diff_log(self, shift=-1, initial=None, span=None)
new = irispie.cum_pct(self, shift=-1, initial=None, span=None)
new = irispie.cum_roc(self, shift=-1, initial=None, span=None)
```


### Class methods changing existing Series objects in-place ###

```
self.cum_diff(shift=-1, initial=None, span=None)
self.cum_diff_log(shift=-1, initial=None, span=None)
self.cum_pct(shift=-1, initial=None, span=None)
self.cum_roc(shift=-1, initial=None, span=None)
```

### Input arguments ###

???+ input "self"
    Time series on whose data a temporal cumulation is calculated (see the
    overview table above).

???+ input "shift"
    A negative integer determining a time lag at which the temporal
    cumulation is calculated. If `shift=None` (or not specified), `shift`
    is set to `-1`.

???+ input "initial"
    Initial value of the cumulative series. If `initial=None` (or not specified),
    the initial value is set to `0` for `diff` and `diff_log`, and to
    `1` for `pct` and `roc`.

???+ input "span"
    Time span on which the values from the original series are cumulated.
    If `span=None` (or not specified), the time span is set to the entire
    time series.


### Returns ###

???+ returns "new"
    New time series with data calculated as temporal cumulation of the
    original data.

???+ returns "self"
    Time series with data replaced by temporal cumulation of the original
    data.

................................................................................
        """
        _catch_invalid_shift(shift, )
        span = Span(None, None, ) if span is None else span
        span = span.resolve(self, )
        direction = span.direction
        factory = _CUMULATIVE_FACTORY[func_name]
        cum_func = factory[direction]
        initial = factory["initial"] if initial is None else initial
        if direction == "forward":
            self._cumulate_forward(shift, cum_func, initial, span, )
        elif direction == "backward":
            self._cumulate_backward(shift, cum_func, initial, span, )

    def _cumulate_forward(self, shift, cum_func, initial, span, /, ) -> None:
        """
        """
        zipped_span = tuple((t, t.shift(shift, )) for t in span)
        zipped_span = tuple((t, sh) for t, sh in zipped_span if sh is not None)
        min_period = min((sh for t, sh in zipped_span), default=span.start_date, )
        initial_span = Span(min_period, span.end_date, )
        change = self.copy()
        self.empty()
        self.set_data(initial_span, initial)
        for t, sh in zipped_span:
            new_data = cum_func(self.get_data(sh, ), change.get_data(t, ), )
            self.set_data(t, new_data)

    def _cumulate_backward(self, shift, cum_func, initial, shifted_backward_range, /, ) -> None:
        """
        """
        orig_range_shifted = Span(self.start_date, self.end_date, -1, )
        orig_range_shifted.shift(shift, )
        shifted_backward_range = shifted_backward_range.resolve(orig_range_shifted, )
        backward_range = shifted_backward_range.copy()
        backward_range.shift(-shift, )
        initial_span = Span(min(shifted_backward_range), backward_range.start_date, )
        orig = self.copy()
        self.empty()
        self.set_data(initial_span, initial, )
        for t, sh in zip(backward_range, shifted_backward_range, ):
            new_data = cum_func(self.get_data(t, ), orig.get_data(t, ), )
            self.set_data(sh, new_data, )

    #]


attributes = (n for n in dir(Inlay) if not n.startswith("_"))
for n in attributes:
    exec(FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__.append(n)


_CUMULATIVE_FACTORY = {
    "diff": {
        "forward": lambda x_past, change_curr: x_past + change_curr,
        "backward": lambda x_future, change_future: x_future - change_future,
        "initial": 0,
    },
    "diff_log": {
        "forward": lambda x_past, change_curr: x_past * exp(change_curr),
        "backward": lambda x_future, change_future: x_future / exp(change_future),
        "initial": 0,
    },
    "pct": {
        "forward": lambda x_past, change_curr: x_past * (1 + change_curr/100),
        "backward": lambda x_future, change_future: x_future / (1 + change_future/100),
        "initial": 1,
    },
    "roc": {
        "forward": lambda x_past, change_curr: x_past * change_curr,
        "backward": lambda x_future, change_future: x_future / change_future,
        "initial": 1,
    },
}


def _catch_invalid_shift(shift: int | str, ):
    """
    """
    if not isinstance(shift, str) and (int(shift) != shift or shift >= 0):
        raise ValueError("Time shift must be a negative integer or a string")


def _roc_from_pct(pct: Real, /, ) -> Real:
    """
    """
    return 1 + pct/100


