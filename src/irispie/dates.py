"""
Time periods and time spans
"""


#[

from __future__ import annotations

from typing import Union, Self, Any, Protocol, TypeAlias, Literal, runtime_checkable
from collections.abc import Iterable, Callable, Iterator
from numbers import (Real, )
import re as _re
import enum as _en
import functools as _ft
import datetime as _dt
import calendar as _ca
import documark as _dm

from .conveniences import copies as _copies
from . import wrongdoings as _wrongdoings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ez_plotly import PlotlyDateAxisModeType

#]


__all__ = (
    "Frequency",
    "yy", "hh", "qq", "mm", "dd", "ii",
    "Span", "EmptySpan", "start", "end",
    "Period", "periods_from_sdmx_strings", "periods_from_iso_strings", "periods_from_python_dates", "periods_from_until", "periods_from_to",
    "period_indexes",
    "YEARLY", "HALFYEARLY", "QUARTERLY", "MONTHLY", "WEEKLY", "DAILY",
    "PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION",
    "refrequent", "convert_to_new_freq",
    "daily_serial_from_ymd",

    "daters_from_sdmx_strings", "daters_from_iso_strings", "daters_from_to",
    "Dater", "Ranger", "EmptyRanger",
    "DATER_CLASS_FROM_FREQUENCY_RESOLUTION",
)


PositionType = Literal["start", "middle", "end", ]


@_dm.reference(
    path=("data_management", "frequencies.md", ),
    categories=None,
)
class Frequency(_en.IntEnum):
    r"""
................................................................................

Time frequencies
=================

Time frequencies are simple integer values that represent the number of time
periods within a year, plus two special frequencies: a so-called "integer"
frequency (for simple numbered observations without relation to calendar time),
and a representation for unknown or unspecified frequencies. For convenience,
the `Frequency` enum provides a set of predefined names for all the time
frequencies available.

The `Frequencies` are classified into regular and
irregular frequencies. Regular frequencies are those that are evenly spaced
within a year no matter the year, while irregular frequencies are those that
vary in the number of periods within a year due to human calendar conventions
and irregularities.


| Integer value | `Frequency` enum       | Regular           | Description
|--------------:|------------------------|:-----------------:|-------------
| 1             | `irispie.YEARLY`       | :material-check:  | Yearly frequency
| 2             | `irispie.HALFYEARLY`   | :material-check:  | Half-yearly frequency
| 4             | `irispie.QUARTERLY`    | :material-check:  | Quarterly frequency
| 12            | `irispie.MONTHLY`      | :material-check:  | Monthly frequency
| 52            | `irispie.WEEKLY`       |                   | Weekly frequency
| 365           | `irispie.DAILY`        |                   | Daily frequency
| 0             | `irispie.INTEGER`      |                   | Integer frequency (numbered observations)
| -1            | `irispie.UNKNOWN`      |                   | Unknown or unspecified frequency


The most often direct use of `Frequencies` in frequency conversion methods, such
as `aggregate` and `disaggregate` for time [`Series`](time_series.md) and whenever a
custom check of time period or time series properties is needed.

................................................................................
    """
    #[

    INTEGER = 0
    YEARLY = 1
    ANNUAL = 1
    HALFYEARLY = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365
    UNKNOWN = -1

    @classmethod
    @_dm.reference(
        category="constructor",
        call_name="Frequency.from_letter",
    )
    def from_letter(
        klass,
        string: str,
        /,
    ) -> Self:
        r"""
................................................................................

==Determine frequency from a letter==

................................................................................
        """
        letter = string.replace("_", "").upper()[0]
        return next( x for x in klass if x.name.startswith(letter) )

    @classmethod
    @_dm.reference(
        category="constructor",
        call_name="Frequency.from_sdmx_string",
    )
    def from_sdmx_string(
        klass,
        sdmx_string: str,
        /,
    ) -> Self:
        r"""
................................................................................

==Determine frequency of an SDMX string==

................................................................................
        """
        try:
            sdmx_string = sdmx_string.strip()
            return next(
                freq for freq, (length, pattern, ) in SDMX_REXP_FORMATS.items()
                if (length is None or len(sdmx_string) == length) and pattern.fullmatch(sdmx_string, )
            )
        except StopIteration:
            raise _wrongdoings.IrisPieCritical(
                f"Cannot determine time frequency from \"{sdmx_string}\"; "
                f"probably not a valid SDMX string"
            )

    @property
    @_dm.reference(category="property", )
    def letter(self, /, ) -> str:
        r"""==Single letter representation of time frequency=="""
        return self.name[0] if self is not self.UNKNOWN else "?"

    @property
    @_dm.reference(category="property", )
    def is_regular(self, /, ) -> bool:
        r"""==True for regular time frequency=="""
        return self in (self.YEARLY, self.HALFYEARLY, self.QUARTERLY, self.MONTHLY, )

    def __str__(self, /, ) -> str:
        return self.name

    #]



_COMPACT_MONTH_STRINGS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _get_compact_year_string(year: int, ) -> str:
    return str(year)[-2:]


Frequency.YEARLY.__doc__ = r"""
................................................................................

==Create a yearly frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.HALFYEARLY.__doc__ = r"""
................................................................................

==Create a half-yearly frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.QUARTERLY.__doc__ = r"""
................................................................................

==Create a quarterly frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.MONTHLY.__doc__ = r"""
................................................................................

==Create a monthly frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.WEEKLY.__doc__ = r"""
................................................................................

==Create a weekly frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.DAILY.__doc__ = r"""
................................................................................

==Create a daily frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""


Frequency.INTEGER.__doc__ = r"""
................................................................................

==Create an integer frequency representation==

See documentation for the time [`Frequency`](frequencies.md).

................................................................................
"""



YEARLY = Frequency.YEARLY
HALFYEARLY = Frequency.HALFYEARLY
QUARTERLY = Frequency.QUARTERLY
MONTHLY = Frequency.MONTHLY
WEEKLY = Frequency.WEEKLY
DAILY = Frequency.DAILY


SDMX_REXP_FORMATS = {
    Frequency.YEARLY: (4, _re.compile(r"\d\d\d\d", ), ),
    Frequency.HALFYEARLY: (7, _re.compile(r"\d\d\d\d-H\d", ), ),
    Frequency.QUARTERLY: (7, _re.compile(r"\d\d\d\d-Q\d", ), ),
    Frequency.MONTHLY: (7, _re.compile(r"\d\d\d\d-\d\d", ), ),
    Frequency.WEEKLY: (8, _re.compile(r"\d\d\d\d-W\d\d", ), ),
    Frequency.DAILY: (10, _re.compile(r"\d\d\d\d-\d\d-\d\d", ), ),
    Frequency.INTEGER: (None, _re.compile(r"\([\-\+]?\d+\),", ), ),
}


BASE_YEAR = 2020


@runtime_checkable
class ResolutionContextProtocol(Protocol, ):
    """
    Context protocol for contextual period resolution
    """
    start_date = ...
    end_date = ...


class ResolutionContext:
    """
    """
    #[

    def __init__(
        self,
        start_date: Period | None = None,
        end_date: Period | None = None,
        /,
    ) -> None:
        """
        """
        self.start_date = start_date
        self.end_date = end_date

    #]


@runtime_checkable
class ResolvableProtocol(Protocol, ):
    """
    Contextual period protocol
    """
    needs_resolve = ...
    def resolve(self, context: ResolutionContextProtocol) -> Any: ...


def _check_periods(first, second, ) -> None:
    if str(type(first)) == str(type(second)):
        return
    message = "Cannot handle periods of different time frequencies in this context"
    raise _wrongdoings.IrisPieError(message, )


def _check_periods_decorator(func: Callable, ) -> Callable:
    def wrapper(*args, **kwargs):
        _check_periods(args[0], args[1], )
        return func(*args, **kwargs, )
    return wrapper


def _remove_blanks(func: Callable,) -> Callable:
    def wrapper(*args, **kwargs, ):
        return func(*args, **kwargs, ).replace(" ", "", )
    return wrapper


class _SpannableMixin:
    """
    """
    #[

    def __rshift__(self, end: Self | None, ) -> Span:
        """
        Period >> Period or Period >> None
        """
        return Span(self, end, 1)

    def __rrshift__(self, start: Self | None, ) -> Span:
        """
        None >> Period
        """
        return Span(start, self, 1)

    def __lshift__(self, start: Self | None, ) -> Span:
        """
        Period << Period or Period << None
        """
        return Span(start, self, -1) 

    def __rlshift__(self, end: Self | None, ) -> Span:
        """
        None << Period
        """
        return Span(self, end, -1) 

    def __pow__(self, len_dir: int, ) -> Span:
        r"""
        Period ** int
        """
        if len_dir == 1 or len_dir == -1:
            return self
        if len_dir > 0:
            return Span(self, self + len_dir - 1, 1, )
        if len_dir < 0:
            return Span(self, self + len_dir + 1, -1, )
        if len_dir == 0:
            return EmptySpan()
        raise ValueError("Invalid span len_dir")

    #]


def _period_constructor_with_ellipsis(
    func: Callable,
    /,
) -> Callable:
    """
    """
    #[
    @_ft.wraps(func, )
    def wrapper(*args, ):
        try:
            index = args.index(Ellipsis, )
            start = func(*args[:index], ) if args[:index] else None
            end = func(*args[index+1:], ) if args[index+1:] else None
            return Span(start, end, )
        except ValueError:
            return func(*args, )
    return wrapper
    #]



@_dm.reference(
    path=("data_management", "periods.md", ),
    categories={
        "constructor": "Creating new time periods",
        "arithmetics_comparison": "Adding, subtracting, and comparing time periods",
        "refrequency": "Converting time periods to different frequencies",
        "conversion": "Converting time periods to different representations",
        "print": "Converting time periods to strings",
    },
)
class Period(
    _SpannableMixin,
    _copies.Mixin,
):
    """
......................................................................

Time periods
=============

A time `Period` represents one single calendar period of time of a certain
frequency (and hence also a certain duration); the time period
[`Frequencies`](frequencies.md) are identified by an integer value.

Time `Periods` are used to timestamp data observations in time
[`Series`](time_series.md) objects, for basic calenadar time arithmetics, and for
creating time [`Spans`](spans.md).

......................................................................
    """
    #[

    frequency = None
    plotly_xaxis_type = None
    needs_resolve = False
    _POSITION_FROM_PLOTLY_MODE = {
        "instant": "start",
        "period": "middle",
    }

    @property
    @_dm.reference(
        category="property",
        call_name="frequency",
    )
    def _frequency():
        """==Time frequency of the time period=="""
        raise NotImplementedError

    @_dm.reference(
        category=None,
        call_name="Time period constructors",
        call_name_is_code=False,
        priority=20,
    )
    def __init__(self, serial: int = 0, ) -> None:
        r"""
......................................................................

Overview of time period constructors:

| Constructor         | Description
|---------------------|-------------
| `irispie.yy`        | Yearly period
| `irispie.hh`        | Half-yearly period
| `irispie.qq`        | Quarterly period
| `irispie.mm`        | Monthly period
| `irispie.dd`        | Daily period
| `irispie.ii`        | Integer period (numbered observations)


### Syntax for creating new time periods ###

    per = yy(year)
    per = hh(year, halfyear)
    per = qq(year, quarter)
    per = mm(year, month)
    per = dd(year, month, day_in_month)
    per = dd(year, None, day_in_year)
    per = ii(number)


### Input arguments ###

???+ input "year"
    Calendar year as integer.

???+ input "halfyear"
    Half-year as integer, 1 or 2.

???+ input "quarter"
    Quarter as integer, 1 to 4.

???+ input "month"
    Month as integer, 1 to 12.

???+ input "day_in_month"
    Day in month as integer, 1 to 31.

???+ input "day_in_year"
    Day in year as integer, 1 to 365 (or 366 in leap years).

???+ input "number"
    Observation number as integer.


### Returns ###


???+ returns "per"
    A `Period` object representing one single time period of a given
    frequency.

......................................................................
        """
        self.serial = int(serial)

    @_dm.reference(
        category=None,
        call_name="Time period arithmetics",
        call_name_is_code=False,
        priority=30,
    )
    def time_period_arithmetics():
        r"""
................................................................................

Time period arithmetics involve operations that can be performed either
between two time periods or between a time period and an integer.
The arithmetic operations include

* **Adding an integer**: Move a time period forward or backward by the
specified number of periods. The integer specifies how many periods of the
respective time frequency to move forward or backward.

* **Subtracting a time period**: Calculate the number of periods between
two time periods. Both periods must be of the same frequency.

* **Subtracting an integer**: Move a time period backward or forward by the
specified number of periods. The integer specifies how many periods of the
respective frequency to move backward or forward.

When performing arithmetic operations involving two time periods, it is necessary 
that both are of the same time frequency. Additionally, some operations involve a 
time period and an integer, such as adding or subtracting a certain number of 
periods represented by an integer.

These operations enable effective management of time period spans and
time-based calculations necessary for scheduling, forecasting, and
historical data analysis in various applications.

................................................................................
        """

    @_dm.reference(
        category=None,
        call_name="Time period comparison",
        call_name_is_code=False,
        priority=20,
    )
    def time_period_comparison():
        r"""
................................................................................

Time period comparison involves comparing two time periods to determine
their relative position in time. The comparison operations include the
following:

| Operation             | Description
|-----------------------|-------------
| `==`                  | Determine whether two time periods are equal.
| `!=`                  | Determine whether two time periods are not equal.
| `<`                   | Determine whether one time period is earlier than another.
| `<=`                  | Determine whether one time period is earlier than or equal to another.
| `>`                   | Determine whether one time period is later than another.
| `>=`                  | Determine whether one time period is later than or equal to another.

The comparison operations require that both time periods are of the same
time [Frequency](frequencies.md).

................................................................................
        """
        raise NotImplementedError

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.from_iso_string",
    )
    def from_iso_string(
        iso_string: str,
        frequency: Frequency = Frequency.DAILY,
    ) -> Self:
        r"""
................................................................................

==Create time period from ISO-8601 string==

Create a time period from an ISO-8601 string representation. The ISO-8601
string format is `yyyy-mm-dd` where `yyyy` is the calendar year, `mm` is
the month of the year, and `dd` is the day of the month, all represented as
integers.

    period = Period.from_iso_string(
        iso_string,
        *,
        frequency=Frequency.DAILY,
    )


### Input arguments ###


???+ input "iso_string"
    ISO-8601 string representation of the time period.

???+ input "frequency"
    Time frequency of the time period; if `None`, a time period of daily
    frequency will be created.


### Returns ###


???+ returns "period"
    Time period object created from the ISO-8601 string.

................................................................................
        """
        year, month, day = iso_string.split("-", )
        period_constructor = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[frequency].from_ymd
        return period_constructor(int(year), int(month), int(day), )

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.from_python_date",
    )
    def from_python_date(
        python_date: _dt.date,
        frequency: Frequency = Frequency.DAILY,
    ) -> Self:
        r"""
................................................................................

==Create time period from Python datetime==

Create a time period from a Python `datetime` object. The time period is
created based on the time frequency specified.

    period = Period.from_python_date(
        python_date,
        *,
        frequency=Frequency.DAILY,
    )


### Input arguments ###


???+ input "python_date"
    Python `datetime.datetime` or `datetime.date` object representing the time
    period.

???+ input "frequency"
    Time frequency of the time period; if `None`, a time period of daily
    frequency will be created.


### Returns ###


???+ returns "period"
    Time period object created from the provided Python `datetime` object.

................................................................................
        """
        year, month, day = python_date.year, python_date.month, python_date.day
        period_constructor = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[frequency].from_ymd
        return period_constructor(int(year), int(month), int(day), )

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.from_sdmx_string",
    )
    def from_sdmx_string(
        sdmx_string: str,
        frequency: Frequency | None = None,
    ) -> Self:
        r"""
................................................................................

==Create time period from SDMX string==

Create a time period from an SDMX string representation. The SDMX string
format is frequency specific and represents the time period as a string
literal.

    period = Period.from_sdmx_string(
        sdmx_string,
        frequency=Frequency.DAILY,
    )

### Input arguments ###

???+ input "sdmx_string"
    SDMX string representation of the time period.

???+ input "frequency"
    Time frequency of the time period. If `None`, the frequency is inferred
    from the SDMX string itself; supplying the frequency is more efficient
    if it is known in advance.

### Returns ###

???+ returns "period"
    Time period object created from the SDMX string.

................................................................................
        """
        frequency = Frequency.from_sdmx_string(sdmx_string, ) if frequency is None else frequency
        return PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[frequency].from_sdmx_string(sdmx_string, )

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.from_ymd",
    )
    def from_ymd(freq: Frequency, *args, ) -> Self:
        r"""
................................................................................

==Create time period from year, month, and day==

Create a time period from the calendar year, month, and day. The time period
is created based on the time frequency specified.

    period = Period.from_ymd(
        freq,
        year,
        month=1,
        day=1,
    )

### Input arguments ###

???+ input "freq"
    Time frequency of the time period.

???+ input "year"
    Calendar year as integer.

???+ input "month"
    Month of the year as integer.

???+ input "day"
    Day of the month as integer.

### Returns ###

???+ returns "period"
    Time period object created from the year, month, and day.

................................................................................
        """
        return PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_ymd(*args, )

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.from_year_segment",
    )
    def from_year_segment(freq: Frequency, *args, ) -> Self:
        r"""
................................................................................

==Create time period from year and segment==

Create a time period from the calendar year and a segment of the year. The
interpretation of the segment as well as the type of the time period created
depends on the time frequency specified.

    period = Period.from_year_segment(
        freq,
        year,
        segment,
    )

### Input arguments ###

???+ input "freq"
    Time frequency of the time period.

???+ input "year"
    Calendar year as integer.

???+ input "segment"
    Segment of the year as integer; the segment can be a half-year, quarter,
    month, or day, depending on the time frequency, `freq`.

### Returns ###

???+ returns "period"
    Time period object created from the year and segment.

................................................................................
        """
        return PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_year_segment(*args, )

    period_from_ymd = from_ymd
    dater_from_ymd = from_ymd

    @staticmethod
    @_dm.reference(
        category="constructor",
        call_name="Period.today",
    )
    def today(freq: Frequency, ) -> Self:
        r"""
................................................................................

==Create time period for today==

Create a time period for the current date. The time period is created based
on the time frequency specified.

    period = Period.today(freq)

### Input arguments ###

???+ input "freq"
    Time frequency of the time period.

### Returns ###

???+ returns "period"
    Time period object for the current date.

................................................................................
        """
        t = _dt.date.today()
        return PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_ymd(t.year, t.month, t.day, )

    @property
    def start(self, /, ) -> Self:
        """
        """
        return self

    start_date = start

    @property
    def end(self, /, ) -> Self:
        """
        """
        return self

    end_date = end

    @property
    @_dm.reference(category="property", )
    def year(self, /, ) -> int:
        r"""==Calendar year of the time period=="""
        ...

    @property
    @_dm.reference(category="property", )
    def segment(self, /, ) -> int:
        r"""==Segment within the calendar year=="""
        ...

    @_dm.reference(category="refrequency", )
    def refrequent(self, new_freq: Frequency, *args ,**kwargs, ) -> Self:
        r"""
................................................................................

==Convert time period to a new frequency==

Convert a time period to a new time frequency by specifying the new
frequency and, optionally, the position of the new time period within the
original time period. The conversion is frequency specific and may require
additional arguments.

    new_period = self.refrequent(
        new_freq,
        *,
        position="start",
    )


### Input arguments ###


???+ input "self"
    Time period to convert to a new time frequency.

???+ input "new_freq"
    New time frequency to which the time period is converted.

???+ input "position"
    Position of the new time period within the original time period. This option
    is effective when the conversion is ambiguous, i.e. from a lower frequency
    period to a higher frequency period. See the position options in
    [`to_ymd`](#to_ymd).


### Returns ###


???+ returns "new_period"
    New time period resulting from the conversion to the new time frequency.

................................................................................
        """
        year, month, day = self.to_ymd(*args, **kwargs, )
        new_class = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
        return new_class.from_ymd(year, month, day, )

    convert_to_new_frequency = refrequent
    convert_to_new_freq = refrequent
    convert = refrequent

    @_dm.reference(category="conversion", )
    def to_ymd(self, **kwargs, ) -> tuple[int, int, int]:
        r"""
................................................................................

==Get year, month, and day of time period==

Get the calendar year, month, and day of the time period as a tuple of
integers.

    year, month, day = self.to_ymd(*, position="start", )


### Input arguments ###


???+ input "self"
    Time period to extract the year, month, and day from.

???+ input "position"
    Position that determins the day of the month and the month of the year
    of time periods with time frequency lower than daily. The position can
    be one of the following:

    * `"start"`: Start of the time period (placed on the 1st day of the
    first month within the original period).

    * `"middle"`: Middle of the time period (placed on the 15th day of the
    middle month within the original period).

    * `"end"`: End of the time period (placed on the last day of the last
    month within the original period).


### Returns ###


???+ returns "year"
    Calendar year of the time period.

???+ returns "month"
    Month of the year of the time period.

???+ returns "day"
    Day of the month of the time period.

................................................................................
        """
        ...

    @_dm.reference(category="print", )
    def to_iso_string(self, **kwargs, ) -> str:
        r"""
................................................................................

==ISO-8601 representation of time period==

Get the ISO-8601 string representation of the time period. The ISO-8601
string format is `yyyy-mm-dd` where `yyyy` is the calendar year, `mm` is
the month of the year, and `dd` is the day of the month, all represented as
integers.

    iso_string = self.to_iso_string(*, position="start", )


### Input arguments ###


???+ input "self"
    Time period to convert to an ISO string.

???+ input "position"
    Position that determines the day of the month and the month of the year
    of time periods with time frequency lower than daily. See the position
    options in [`to_ymd`](#to_ymd).


### Returns ###


???+ returns "iso_string"
    ISO-8601 string representation of the time period.

................................................................................
        """
        year, month, day = self.to_ymd(**kwargs, )
        return f"{year:04g}-{month:02g}-{day:02g}"

    @_dm.reference(category="print", )
    def to_sdmx_string(self, /, ) -> str:
        r"""
................................................................................

==SDMX representation of time period==

The SDMX string representation of the time periods is a standardized format
used in statistical data exchange. The SDMX string format is frequency
specific:

Time frequency | SDMX format   | Example
---------------|---------------|--------
Yearly         | `yyyy`        | `2030`
Half-yearly    | `yyyy-Hh`     | `2030-H1`
Quarterly      | `yyyy-Qq`     | `2030-Q1`
Monthly        | `yyyy-mm`     | `2030-01`
Weekly         | `yyyy-Www`    | `2030-W01`
Daily          | `yyyy-mm-dd`  | `2030-01-01`
Integer        | `(n)`         | `(1)`

where lowercase letters represent the respective time period components
(integer values) and uppercase letters are literals.


    sdmx_string = self.to_sdmx_string()


### Input arguments ###


???+ input "self"
    Time period to convert to an SDMX string.


### Returns ###


???+ returns "sdmx_string"
    SDMX string representation of the time period.

................................................................................
        """
        ...

    @_dm.reference(category="print", )
    def to_compact_string(self, /, ) -> str:
        r"""
................................................................................

==Compact representation of time period==

The compact string format is frequency specific:

Time frequency | Compact format   | Example
---------------|------------------|--------
Yearly         | `yyY`            | `30Y`
Half-yearly    | `yyHh`           | `30H1`
Quarterly      | `yyQq`           | `30Q1`
Monthly        | `yyMmm`          | `30M01`
Weekly         | `yyWww`          | `30W01`
Daily          | `yymmmdd`        | `30Jan01`
Integer        | `(n)`            | `(1)`

where lowercase letters represent the respective time period components
(integer values) and uppercase letters are literals.


    compact_string = self.to_compact_string()


### Input arguments ###


???+ input "self"
    Time period to convert to a compact string.


### Returns ###


???+ returns "compact_string"
    Compact string representation of the time period.

................................................................................
        """
        ...

    @_dm.reference(category="conversion", )
    def to_python_date(
        self,
        /,
        position: PositionType = "start",
    ) -> _dt.date:
        r"""
................................................................................


==Convert time period to Python date object==


Convert a time period to a Python date object. The date object is created based
on the year, month, and day of the time period.


    date = self.to_python_date(
        position="middle",
    )


### Input arguments ###


???+ input "self"
    Time period to convert to a Python date object.

???+ input "position"
    Position that determines the day of the month and the month of the year
    of time periods with time frequency lower than daily. The position can
    be one of the following:

    * `"start"`: Start of the time period (placed on the 1st day of the
    first month within the original period).

    * `"middle"`: Middle of the time period (placed on the 15th day of the
    middle month within the original period).

    * `"end"`: End of the time period (placed on the last day of the last
    month within the original period).


### Returns ###


???+ returns "date"
    Python date object representing the time period.


................................................................................
        """
        return _dt.date(*self.to_ymd(position=position, ))

    def to_plotly_date(
        self,
        mode: PlotlyDateAxisModeType = "period",
    ) -> _dt.date:
        position = self._POSITION_FROM_PLOTLY_MODE[mode]
        return self.to_python_date(position=position, )

    def to_plotly_edge_before(
        self,
        mode: PlotlyDateAxisModeType = "period",
    ) -> _dt.date:
        r"""
        """
        return {
            "period": self.to_python_date(position="start", ),
            "instant": (self - 1).to_python_date(position="middle", ),
        }[mode]

    def to_plotly_edge_after(
        self,
        mode: PlotlyDateAxisModeType = "period",
    ) -> _dt.date:
        r"""
        """
        return {
            "period": self.to_python_date(position="end", ),
            "instant": self.to_python_date(position="middle", ),
        }[mode]

    @_dm.reference(category="information", )
    def get_distance_from_origin(self, ) -> int:
        r"""
................................................................................

==Get distance from origin time period==

Get the distance of the time period from the origin time period. The origin time
period is currently set to the beginning of year 2020 for all calendar periods,
and to 0 for integer periods.

    distance = self.get_distance_from_origin()

### Returns ###

???+ returns "distance"
    Distance of the `self` time period from the origin time period.

................................................................................
        """
        return self.serial - self.origin

    @_dm.no_reference
    def resolve(self, context: ResolutionContextProtocol) -> Self:
        return self

    def __bool__(self) -> bool:
        return not self.needs_resolve

    def __len__(self):
        return 1

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.to_sdmx_string()

    _databox_repr = __str__

    def __format__(self, *args) -> str:
        str_format = args[0] if args else ""
        return ("{period_str:"+str_format+"}").format(period_str=self.__str__())

    def __iter__(self) -> Iterator[Self]:
        yield self

    def __hash__(self, /, ) -> int:
        return hash((int(self.serial), hash(self.frequency), ))

    def __add__(self, other: int, ) -> Self:
        r"""
................................................................................

==Add integer to time period==

Add an integer value to a time period, moving it forward by the number of
periods (if positive) or backward (if negative).

    new_period = period + k
    new_period = k + period


### Input arguments ###


???+ input "period"
    Time period to which the integer is added.

???+ input "k"
    Integer value to add to the time period. Positive values move the
    period forward, while negative values move it backward. The addition is
    frequency specific.


### Returns ###


???+ returns "new_period"
    New time period resulting from the addition of the integer value.


### See also ###

* [Time period arithmetics](#time-period-arithmetics)

................................................................................
        """
        return type(self)(self.serial + int(other))

    __radd__ = __add__

    def __sub__(self, other: Self | int) -> Self | int:
        r"""
................................................................................

==Subtract time period or integer from time period==

Subtraction operations can be performed between two time periods or between a 
time period and an integer. These operations are crucial for calculating the 
distance between periods or adjusting a period's position in time.

* **Subtracting a time period**: Calculate the number of periods between
two time periods. Both periods must be of the same frequency.

* **Subtracting an integer**: Move a time period backward or forward by the
specified number of periods. The integer specifies how many periods of the
respective frequency to move backward or forward.


    result = self - other
    new_period = self - k


### Input arguments ###


???+ input "self"
    The reference time period from which `other` or `k` is subtracted.

???+ input "other"
    The time period to subtract from `self`. The `result` is the number of
    periods between `self` and `other`. The subtraction is frequency specific.

???+ input "k"
    Integer value to subtract from `self`. Positive values move the period
    backward, while negative values move it forward. The subtraction is
    frequency specific.


### Returns ###


???+ returns "result"
    The number of periods between two time periods if `other` is a `Period`.

???+ returns "new_period"
    A new `Period` object representing the time period shifted backward by the 
    specified number of periods if `other` is an integer.

................................................................................
    """
        if _is_period(other, ):
            return self._sub_period(other)
        else:
            return self.__add__(-int(other))

    @_check_periods_decorator
    def _sub_period(self, other: Self) -> int:
        return self.serial - other.serial

    def __index__(self):
        return self.serial

    def __eq__(self, other: Self, /, ) -> bool:
        r"""
................................................................................

==True if time period is equal to another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial == other.serial

    def __ne__(self, other: Self, /, ) -> bool:
        r"""
................................................................................

==True if time period is not equal to another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial != other.serial

    def __lt__(self, other: Self, /, ) -> bool:
        r"""
................................................................................

==True if time period is earlier than another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial < other.serial

    def __le__(self, other: Self, /, ) -> bool: 
        r"""
................................................................................

==True if time period is earlier than or equal to another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial <= other.serial

    def __gt__(self, other: Self, /, ) -> bool:
        r"""
................................................................................

==True if time period is later than another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial > other.serial

    def __ge__(self, other: Self, /, ) -> bool:
        r"""
................................................................................

==True if time period is later than or equal to another time period==

See documentation for [time period comparison](#time-period-comparison).

................................................................................
        """
        _check_periods(self, other, )
        return self.serial >= other.serial

    @_dm.reference(
        category="arithmetics_comparison",
    )
    def shift(
        self,
        by: int | str = -1,
    ) -> Self:
        r"""
................................................................................

==Shift time period by a number of periods==

Shift a time period forward or backward by the specified number of periods. 

    self.shift(k)


### Input arguments ###


???+ input "self"
    Time period to shift forward or backward.

???+ input "k"
    Integer value specifying the number of periods to move the time period.
    Positive values move the period forward, while negative values move it
    backward. The shift is frequency specific.


### Returns ###


Returns no value; the time period is modified in place.

................................................................................
        """
        match by:
            case "yoy":
                return self - self.frequency.value
            case "boy" | "soy":
                return self.create_soy()
            case "eopy":
                return self.create_eopy()
            case "tty":
                return self.create_tty()
            case _:
                return self + by

    #]


class IntegerPeriod(Period, ):
    """
    """
    #[

    frequency = Frequency.INTEGER
    plotly_xaxis_type = "linear"
    needs_resolve = False
    half_period = 0.5
    origin = 0

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> IntegerPeriod:
        sdmx_string = sdmx_string.strip().removeprefix("(").removesuffix(")")
        return klass(int(sdmx_string))

    def to_sdmx_string(self, /, ) -> str:
        return f"({self.serial})"

    to_compact_string = to_sdmx_string

    def __repr__(self) -> str:
        return f"ii({self.serial})"

    def to_plotly_date(self, *args, **kwargs, ) -> Real:
        return self.serial

    def to_plotly_edge_before(self, *args, **kwargs, ) -> Real:
        return self.serial - self.half_period

    def to_plotly_edge_after(self, *args, **kwargs, ) -> Real:
        return self.serial + self.half_period

    @classmethod
    def from_year_segment(
        klass,
        year: Any,
        segment: int = 0,
    ) -> Self:
        r"""
        """
        serial = int(segment)
        return klass(serial)

    from_year_period = from_year_segment

    #]


class DailyPeriod(Period, ):
    """
    """
    #[

    frequency: Frequency = Frequency.DAILY
    plotly_xaxis_type = "date"
    needs_resolve = False
    origin = _dt.date(BASE_YEAR, 1, 1).toordinal()

    @classmethod
    def from_ymd(klass: type, year: int, month: int=1, day: int=1) -> Self:
        serial = daily_serial_from_ymd(year, month, day, )
        return klass(serial)

    @classmethod
    def from_year_segment(
        klass,
        year: int,
        segment: int = 1,
    ) -> Self:
        r"""
        """
        boy_serial = _dt.date(year, 1, 1).toordinal()
        serial = boy_serial + int(segment) - 1
        return klass(serial)

    from_year_period = from_year_segment

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> Self:
        year, month, day, *_ = sdmx_string.split("-")
        return klass.from_ymd(int(year), int(month), int(day))

    @classmethod
    def from_iso_string(klass, iso_string: str, ) -> Self:
        """
        """
        year, month, day, *_ = iso_string.split("-", )
        return klass.from_ymd(int(year), int(month), int(day), )

    @property
    def year(self, /, ) -> int:
        return _dt.date.fromordinal(self.serial).year

    @property
    def month(self, /, ) -> int:
        return _dt.date.fromordinal(self.serial).month

    @property
    def day(self, /, ) -> int:
        return _dt.date.fromordinal(self.serial).day

    @property
    def segment(self, /, ) -> int:
        return self.to_year_segment()[1]

    @property
    def period(self, /, ) -> int:
        return self.segment

    def to_sdmx_string(self, /, **kwargs) -> str:
        year, month, day = self.to_ymd()
        return f"{year:04g}-{month:02g}-{day:02g}"

    def to_compact_string(self, /, **kwargs) -> str:
        year, month, day = self.to_ymd()
        year_string = _get_compact_year_string(year)
        month_string = _COMPACT_MONTH_STRINGS[month-1]
        return f"{year_string}{month_string}{day:02g}"

    def to_year_segment(self) -> tuple[int, int]:
        boy_serial = _dt.date(_dt.date.fromordinal(self.serial).year, 1, 1)
        per = self.serial - boy_serial + 1
        year = _dt.date.fromordinal(self.serial).year
        return year, per

    def to_ymd(self, /, **kwargs) -> tuple[int, int, int]:
        py_date = _dt.date.fromordinal(self.serial)
        return py_date.year, py_date.month, py_date.day

    def get_year(self, /, ) -> int:
        return _dt.date.fromordinal(self.serial).year

    @_remove_blanks
    def __repr__(self) -> str:
        return f"dd{self.to_ymd()}"

    def create_soy(self, ) -> Self:
        year = self.get_year()
        serial = _dt.date(year, 1, 1).toordinal()
        return type(self)(serial)

    def create_som(self, ) -> Self:
        year, month, _ = self.to_ymd()
        serial = _dt.date(year, month, 1).toordinal()
        return type(self)(serial)

    def create_eoy(self, ) -> Self:
        year = self.get_year()
        serial = _dt.date(year, 12, 31).toordinal()
        return type(self)(serial)

    def create_eopy(self, ) -> Self:
        year = self.get_year()
        serial = _dt.date(year-1, 12, 31).toordinal()
        return type(self)(serial)

    def create_eopm(self, ) -> Self:
        return self.create_som() - 1

    def create_tty(self, ) -> Self | None:
        _, seg = self.to_year_segment()
        return self - 1 if seg > 1 else None

    def to_daily(self, /, **kwargs, ) -> Self:
        return self

    #]


def _serial_from_ysf(year: int, per: int, freq: int) -> int:
    return int(year)*int(freq) + int(per) - 1


class RegularPeriodMixin:
    """
    """
    #[

    plotly_xaxis_type = "date"

    @classmethod
    def from_year_segment(
            klass: type,
            year: int,
            per: int | str = 1,
        ) -> Self:
        per = per if per != "end" else klass.frequency.value
        new_serial = _serial_from_ysf(year, per, klass.frequency.value)
        return klass(new_serial, )

    from_year_period = from_year_segment

    @classmethod
    def from_ymd(klass, year: int, month: int=1, day: int=1, ) -> Self:
        return klass.from_year_segment(year, klass.month_to_segment(month, ), )

    @classmethod
    def from_iso_string(klass, iso_string: str, ) -> Self:
        """
        """
        year, month, day, *_ = iso_string.split("-", )
        return klass.from_ymd(int(year), int(month), int(day), )

    @property
    def year(self, ) -> int:
        return self.to_year_segment()[0]

    @property
    def segment(self, ) -> int:
        return self.to_year_segment()[1]

    @property
    def period(self, ) -> int:
        return self.segment

    def to_year_segment(self) -> tuple[int, int]:
        return self.serial//self.frequency.value, self.serial%self.frequency.value+1

    def get_year(self) -> int:
        return self.to_year_segment()[0]

    def to_ymd(
        self, 
        position: PositionType = "start",
    ) -> tuple[int, int, int]:
        year, per = self.to_year_segment()
        month, day = self._MONTH_DAY_RESOLUTION[position][per]
        if day is None:
            _, day = _ca.monthrange(year, month)
        return year, month, day

    def __str__(self, /, ) -> str:
        return self.to_sdmx_string()

    def create_soy(self, ) -> Self:
        year, *_ = self.to_year_segment()
        return self.from_year_segment(year, 1)

    create_boy = create_soy

    def create_eoy(self, ) -> Self:
        year, *_ = self.to_year_segment()
        return self.from_year_segment(year, "end")

    def create_eopy(self, ) -> Self:
        year, *_ = self.to_year_segment()
        return self.from_year_segment(year-1, self.frequency.value)

    def create_tty(self, ) -> Self | None:
        _, seg = self.to_year_segment()
        return self - 1 if seg > 1 else None

    def to_daily(
        self,
        /,
        position: PositionType = "start"
    ) -> DailyPeriod:
        try:
            return DailyPeriod.from_ymd(*self.to_ymd(position=position, ), )
        except:
            raise IrisPieCritical("Cannot convert period to daily period.")

    #]


class YearlyPeriod(RegularPeriodMixin, Period, ):
    """
    """
    #[

    frequency: Frequency = Frequency.YEARLY
    needs_resolve: bool = False
    origin = _serial_from_ysf(BASE_YEAR, 1, Frequency.YEARLY)
    _MONTH_DAY_RESOLUTION = {
        "start": {1: (1, 1)},
        "middle": {1: (6, 30)},
        "end": {1: (12, 31)},
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> YearlyPeriod:
        return klass(int(sdmx_string.strip()))

    def to_sdmx_string(self, /, ) -> str:
        return f"{self.get_year():04g}"

    def to_compact_string(self, /, ) -> str:
        year_string = _get_compact_year_string(self.get_year())
        return f"{year_string}Y"

    @_remove_blanks
    def __repr__(self) -> str: return f"yy({self.get_year()})"

    @staticmethod
    def month_to_segment(month: int, ) -> int:
        return 1

    #]


class HalfyearlyPeriod(RegularPeriodMixin, Period, ):
    #[
    frequency: Frequency = Frequency.HALFYEARLY
    needs_resolve: bool = False
    origin = _serial_from_ysf(BASE_YEAR, 1, Frequency.HALFYEARLY)
    _MONTH_DAY_RESOLUTION = {
        "start": {1: (1, 1), 2: (7, 1)},
        "middle": {1: (3, 15), 2: (9, 15)},
        "end": {1: (6, 30), 2: (12, 31)}
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> HalfyearlyPeriod:
        year, halfyear = sdmx_string.strip().split("-H")
        return klass.from_year_segment(int(year), int(halfyear))

    def to_sdmx_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        return f"{year:04g}-{self.frequency.letter}{per:1g}"

    def to_compact_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        year_string = _get_compact_year_string(year)
        return f"{year_string}{self.frequency.letter}{per:1g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"hh{self.to_year_segment()}"

    def get_month(
        self,
        /,
        position: PositionType = "start",
    ) -> int:
        _, per = self.to_year_segment()
        return month_resolution[position][per]

    @staticmethod
    def month_to_segment(month: int, ) -> int:
        return 1+((month-1)//6)
    #]


class QuarterlyPeriod(RegularPeriodMixin, Period, ):
    """
    """
    #[

    frequency: Frequency = Frequency.QUARTERLY
    needs_resolve: bool = False
    origin = _serial_from_ysf(BASE_YEAR, 1, Frequency.QUARTERLY)
    _MONTH_DAY_RESOLUTION = {
        "start": {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)},
        "middle": {1: (2, 15), 2: (5, 15), 3: (8, 15), 4: (11, 15)},
        "end": {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)},
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> QuarterlyPeriod:
        year, quarter = sdmx_string.strip().split("-Q")
        return klass.from_year_segment(int(year), int(quarter))

    def to_sdmx_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        return f"{year:04g}-{self.frequency.letter}{per:1g}"

    def to_compact_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        year_string = _get_compact_year_string(year)
        return f"{year_string}{self.frequency.letter}{per:1g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"qq{self.to_year_segment()}"

    @staticmethod
    def month_to_segment(month: int, ) -> int:
        return 1+((month-1)//3)

    #]


class MonthlyPeriod(RegularPeriodMixin, Period, ):
    #[
    frequency: Frequency = Frequency.MONTHLY
    needs_resolve: bool = False
    origin = _serial_from_ysf(BASE_YEAR, 1, Frequency.MONTHLY)
    _MONTH_DAY_RESOLUTION = {
        "start": { m: (m, 1) for m in range(1, 13) },
        "middle": { m: (m, 15) for m in range(1, 13) },
        "end": { m: ((m, _ca.monthrange(1970, m)[1]) if m != 2 else (m, None)) for m in range(1, 13) },
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str, ) -> MonthlyPeriod:
        year, month = sdmx_string.strip().split("-")
        return klass.from_year_segment(int(year), int(month))

    def to_sdmx_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        return f"{year:04g}-{per:02g}"

    def to_compact_string(self, /, ) -> str:
        year, per = self.to_year_segment()
        year_string = _get_compact_year_string(year)
        return f"{year_string}{self.frequency.letter}{per:02g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"mm{self.to_year_segment()}"

    @staticmethod
    def month_to_segment(month: int, ) -> int:
        return month
    #]


class UnknownPeriod:
    """
    """
    #[

    def from_sdmx_string(sdmx_string: str, ) -> None:
        return None

    #]


yy = _period_constructor_with_ellipsis(YearlyPeriod.from_year_segment, )
yy.__doc__ = r"""
................................................................................

==Create a yearly-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
"""
yy.__name__ = "irispie.yy"
yy = _dm.reference(category="constructor", )(yy)


hh = _period_constructor_with_ellipsis(HalfyearlyPeriod.from_year_segment, )
hh.__doc__ = r"""
................................................................................

==Create a half-yearly-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
"""
hh.__name__ = "irispie.hh"
hh = _dm.reference(category="constructor", )(hh)


qq = _period_constructor_with_ellipsis(QuarterlyPeriod.from_year_segment)
qq.__doc__ = r"""
................................................................................

==Create a quarterly-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
"""
qq.__name__ = "irispie.qq"
qq = _dm.reference(category="constructor", )(qq)


mm = _period_constructor_with_ellipsis(MonthlyPeriod.from_year_segment)
mm.__doc__ = r"""
................................................................................

==Create a monthly-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
"""
mm.__name__ = "irispie.mm"
mm = _dm.reference(category="constructor", )(mm)


ii = _period_constructor_with_ellipsis(IntegerPeriod)
ii.__doc__ = r"""
................................................................................

==Create an integer-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
"""
ii.__name__ = "irispie.ii"
ii = _dm.reference(category="constructor", )(ii)


@_dm.reference(
    category="constructor",
    call_name="irispie.dd",
)
def dd(year: int, month: int | None, day: int) -> DailyPeriod:
    r"""
................................................................................

==Create a daily-frequency time period or time span==

See documentation for the [`Period` constructors](#time-period-constructors)
and the [`Span` constructors](spans.md).

................................................................................
    """
    if month is None:
        return DailyPeriod.from_year_segment(year, day)
    else:
        return DailyPeriod.from_ymd(year, month, day)


for n in ("yy", "hh", "qq", "mm", "dd", "ii", ):
    setattr(Period, n, locals()[n], )


def daily_serial_from_ymd(year: int, month: int, day: int, ) -> int:
    return _dt.date(year, month, day).toordinal()


@_dm.reference(
    path=("data_management", "spans.md", ),
    categories={
        "constructor": "Creating new time spans",
        "arithmetics": "Arithmetic operations on time spans",
        "manipulation": "Manipulating time spans",
        "print": "Converting time spans to strings",
        "property": None,
    },
)
class Span(
    _copies.Mixin,
):
    """
................................................................................

Time spans
============

Time spans represent a range of time periods of the same time frequency,
from a start period to an end period (possibly with a step size other than
1), going either forward or backward.

................................................................................
    """
    #[

    @_dm.reference(
        category="constructor",
        call_name="Span",
        priority=20,
    )
    def __init__(
        self,
        from_per: Period | None = None,
        until_per: Period | None = None,
        step: int = 1,
    ) -> None:
        r"""
................................................................................

==Create a new time span==


### Using the `Span` constructor ###

    span = Span(start_per, end_per, step=1)


### Shorthand using the `>>` and `<<` operators ###

    span = start_per >> end_per #[^1]
    span = end_per << start_per #[^2]

1. Equivalent to `Span(start_per, end_per, step=1)`
2. Equivalent to `Span(end_per, start_per, step=-1)`

................................................................................
        """
        if step > 0:
            default_from_per = start
            default_until_per = end
        else:
            default_from_per = end
            default_until_per = start
        self._start = from_per if from_per is not None else default_from_per
        self._end = until_per if until_per is not None else default_until_per
        self._step = step
        self.needs_resolve = self._start.needs_resolve or self._end.needs_resolve
        if not self.needs_resolve:
            _check_periods(from_per, until_per)

    @classmethod
    def encompassing(klass, *args, ) -> Self:
        """
        """
        return get_encompassing_span(*args, )[0]

    @property
    @_dm.reference(category="property", )
    def start(self):
        """==Start period of the time span=="""
        return self._start

    start_date = start

    @property
    @_dm.reference(category="property", )
    def end(self):
        """==End period of the time span=="""
        return self._end

    end_date = end

    @property
    @_dm.reference(category="property", )
    def step(self):
        """==Step size of the time span=="""
        return self._step

    @property
    def _class(self):
        return type(self._start) if not self.needs_resolve else None

    @property
    @_dm.reference(category="property", )
    def direction(self, ) -> Literal["forward", "backward", ]:
        """==Direction of the time span=="""
        return "forward" if self._step > 0 else "backward"

    @property
    def _serials(self) -> range | None:
        return range(self._start.serial, self._end.serial+_sign(self._step), self._step) if not self.needs_resolve else None

    # @property
    # def needs_resolve(self) -> bool:
        # return bool(self._start and self._end)

    @property
    @_dm.reference(category="property", )
    def frequency(self) -> Frequency:
        """==Frequency of the time span=="""
        return self._class.frequency

    @_dm.reference(category="manipulation", )
    def reverse(self, ) -> None:
        r"""
................................................................................

==Reverse the time span==

Reverses the direction of the time span, so that the start period becomes
the end period and vice versa.

    self.reverse()


### Input arguments ###

???+ input "self"
    The time span to be reversed.


### Returns ###

The time span is reversed in place.

................................................................................
        """
        self._start, self._end = self._end, self._start
        self._step = -self._step

    def reversed(self, ) -> Self:
        """
        """
        new = self.copy()
        new.reverse()
        return new

    @_dm.reference(category="manipulation", )
    def shift_end(
        self,
        by: int,
        /,
    ) -> Self:
        r"""
................................................................................

==Shift the end of the time span==

Shifts the end of the time span by a specified number of periods. This
operation modifies the end boundary of the time span, effectively changing its
length. Adjusting the end allows for extension or reduction of the span
depending on the direction and magnitude of the shift.

    self.shift_end(by)

### Input arguments ###

???+ input "self"
    The time span within which the end will be shifted.

???+ input "by"
    The number of periods by which the end will be shifted. This can be
    positive (to extend the span by moving the end forward) or negative
    (to reduce the span by moving the end backward).

### Returns ###

???+ returns "None"
    This method modifies `self` in-place and does not return a value.

................................................................................
        """
        self._end += by

    @_dm.reference(category="manipulation", )
    def shift_start(
        self,
        by: int,
        /,
    ) -> Self:
        r"""
................................................................................

==Shift the start period of the time span==

Shifts the start period of the time span by a specified number of periods. This
operation adjusts the start boundary of the time span, effectively changing its
length depending on the direction and magnitude of the shift.

    self.shift_start(by)

### Input arguments ###

???+ input "self"
    The time span within which the start period will be shifted.

???+ input "by"
    The number of periods by which the start period will be shifted. This can be
    positive (to move the start period forward, reducing the span length) or
    negative (to move it backward, increasing the span length).

### Returns ###

This method modifies the object in place and does not return a value.

................................................................................
        """
        self._start += by

    def to_plotly_dates(self, *args, **kwargs, ) -> tuple[str]:
        return tuple(t.to_plotly_date(*args, **kwargs, ) for t in self)

    @_dm.reference(category="print", )
    def to_iso_strings(self, *args, **kwargs, ) -> tuple[str]:
        r"""
................................................................................

==Convert time span periods to ISO-8601 representations==

Converts each period within the time span to an ISO-8601 string format. 

    iso_strings = self.to_iso_strings(*, position="start", )


### Input arguments ###


???+ input "self"
    The time span whose periods are to be converted to ISO-8601 strings.

???+ input "position"
    The position within each period to use when converting to an ISO-8601
    date string. See the documentation for the
    [`to_ymd`](periods.md#to_ymd) method of time [`Periods`](periods.md).


### Returns ###


???+ returns "iso_strings"
    A tuple of ISO-8601 date strings representing each period in the time
    span.

................................................................................
        """
        return tuple(t.to_iso_string(*args, **kwargs, ) for t in self)

    @_dm.reference(category="print", )
    def to_sdmx_strings(self, *args, **kwargs, ) -> tuple[str]:
        r"""
................................................................................

==Convert time span periods to SDMX representations==

Converts each period within the time span to a SDMX string format. 

    sdmx_strings = self.to_sdmx_strings()


### Input arguments ###


???+ input "self"
    The time span whose periods are to be converted to SDMX strings.


### Returns ###


???+ returns "sdmx_strings"
    A tuple of SDMX strings representing each period in the time span.


### See also ###

* [`to_sdmx_string`](periods.md#to_sdmx_string) method of time [`Periods`](periods.md)

................................................................................
        """
        return tuple(t.to_sdmx_string(*args, **kwargs, ) for t in self)

    @_dm.reference(category="print", )
    def to_compact_strings(self, *args, **kwargs, ) -> tuple[str]:
        r"""
        """
        return tuple(t.to_compact_string(*args, **kwargs, ) for t in self)

    def to_python_dates(self, *args, **kwargs, ) -> tuple[_dt.date]:
        return tuple(t.to_python_date(*args, **kwargs, ) for t in self)

    def __len__(self) -> int|None:
        return len(self._serials) if not self.needs_resolve else None

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        start_rep = self._start.__repr__()
        end_rep = self._end.__repr__()
        step_rep = f", {self._step}" if self._step!=1 else ""
        return f"Span({start_rep}, {end_rep}{step_rep})"

    @_dm.reference(
        category="arithmetics",
        call_name="+",
    )
    def __add__(self, offset: int) -> Self:
        r"""
................................................................................

==Add an offset to the time span==

Shifts both the start and end of the time span by a specified number of periods.
This method is used to adjust the entire span forward or backward in time. It can be 
used either by adding the offset to the span (`span + offset`) or the offset to the 
span (`offset + span`), effectively creating a new time span that begins and ends
earlier or later than the original.

    new_span = self + offset
    new_span = offset + self

### Input arguments ###

???+ input "self"
    The time span to be adjusted.

???+ input "offset"
    The number of periods by which to shift the time span. This must be an integer,
    where positive values indicate a forward shift and negative values indicate a 
    backward shift.

### Returns ###

???+ returns "new_span"
    A new `Span` object representing the time span shifted by the specified number of
    periods.

................................................................................
        """
        return type(self)(self._start+offset, self._end+offset, self._step)

    __radd__ = __add__

    def __rshift__(
        self,
        step: int,
    ) -> Self:
        r"""
        """
        if step < 0:
            raise ValueError("Step must be positive when using the >> operator.")
        return type(self)(self._start, self._end, step, )

    def __lshift__(
        self,
        step: int,
    ) -> Self:
        r"""
        """
        if step > 0:
            raise ValueError("Step must be negative when using the << operator.")
        return type(self)(self._start, self._end, step, )

    @_dm.reference(
        category="arithmetics",
        call_name="-",
    )
    def __sub__(self, offset: Period | int, ) -> range | Self:
        r"""
................................................................................

==Subtract an offset or a Period from the time span==

Adjusts the time span by shifting it backward by a specified number of periods or 
computes a range of integers when a `Period` is subtracted. This method can be used 
to shift the entire span backward in time by an integer offset or to calculate the 
distance between each period in the span and a given `Period`.

    new_span = self - offset
    range_result = self - period


### Input arguments ###


???+ input "self"
    The time span from which the offset or a `Period` is to be subtracted.

???+ input "other"
    If an integer, the number of periods by which to shift the time span backward.
    If a `Period`, a specific period used to calculate the difference in periods 
    between this `Period` and each period within the time span.


### Returns ###


???+ returns "new_span" if `other` is an integer
    A new `Span` object representing the time span shifted backward by the specified 
    number of periods.

???+ returns "range_result" if `other` is a `Period`
    A standard range object containing the distances in periods from each period within
    the span to the specified `Period`.

................................................................................
        """
        if _is_period(offset, ):
            return range(self._start-offset, self._end-offset, self._step) if not self.needs_resolve else None
        else:
            return type(self)(self._start-offset, self._end-offset, self._step)

    def __rsub__(self, ather: Period, ) -> range|None:
        if _is_period(other, ):
            return range(other-self._start, other-self._end, -self._step) if not self.needs_resolve else None
        else:
            return None

    def __iter__(self, /, ) -> Iterable:
        return (self._class(x) for x in self._serials) if not self.needs_resolve else None

    def __getitem__(self, i: int, /, ) -> Period | tuple[Period] | None:
        if isinstance(i, int):
            return self._class(self._serials[i]) if not self.needs_resolve else None
        elif isinstance(i, slice):
            indexes = range(*i.indices(len(self)))
            return tuple(t for i, t in enumerate(self) if i in indexes)

    def resolve(self, context: ResolutionContextProtocol, /, ) -> Self:
        resolved_start = self._start if self._start else self._start.resolve(context, )
        resolved_end = self._end if self._end else self._end.resolve(context, )
        return type(self)(resolved_start, resolved_end, self._step, )

    @_dm.reference(category="manipulation", )
    def shift(self, by: int) -> None:
        r"""
................................................................................

==Shift the entire time span==

Shifts the entire time span forward or backward by a specified number of
periods. This method adjusts both the start and end of the span simultaneously,
keeping the length of the span unchanged but moving it entirely to a new
position in the timeline.

    self.shift(by)

### Input arguments ###

???+ input "self"
    The time span that will be shifted along the timeline.

???+ input "by"
    The number of periods to shift the time span. Positive values shift the span 
    forward, while negative values shift it backward.

### Returns ###

???+ returns "None"
    This method modifies `self` in-place and does not return a value.


................................................................................
        """
        self._start += by
        self._end += by

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __eq__(self, other: Self) -> bool:
        return self._start==other._start and self._end==other._end and self._step==other._step

    def __bool__(self) -> bool:
        return not self.needs_resolve

    #]


class EmptySpan:
    """
    """
    #[
    start_date = None
    end_date = None
    step = None
    needs_resolve = False

    def __new__(
        klass,
        *args,
        **kwargs,
    ) -> Self:
        """
        """
        if not hasattr(klass, "_instance"):
            klass._instance = super().__new__(klass, *args, **kwargs, )
        return klass._instance

    def __iter__(self, ) -> Iterable:
        return iter(())

    def __len__(self, ) -> int:
        return 0

    def shift(self, *args, **kwargs, ) -> None:
        pass

    #]




def _sign(x: Real, ) -> int:
    return 1 if x>0 else (0 if x==0 else -1)


def period_indexes(periods: Iterable[Period | None], base: Period) -> Iterable[int]:
    """
    """
    return (
        (t - base) if t is not None else None
        for t in periods
    )


class ContextualPeriod(
    Period,
    _SpannableMixin,
):
    """
    Periods with context dependent resolution
    """
    #[
    needs_resolve = True

    def __init__(self, resolve_from: str, offset: int=0) -> None:
        self._resolve_from = resolve_from
        self._offset = offset

    def __add__(self, offset: int) -> None:
        return type(self)(self._resolve_from, self._offset+offset, )

    def __sub__(self, offset: int) -> None:
        return type(self)(self._resolve_from, self._offset-offset, )

    def __str__(self) -> str:
        return "<>." + self._resolve_from.replace("_date", "") + (f"{self._offset:+g}" if self._offset else "")

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return False

    def resolve(self, context: ResolutionContextProtocol) -> Period:
        return getattr(context, self._resolve_from) + self._offset
    #]


start = ContextualPeriod("start_date", )
end = ContextualPeriod("end_date", )


def resolve_period_or_integer(input_period: Any, /, ) -> Period:
    """
    Convert non-dater to integer dater
    """
    return (
        IntegerPeriod(int(input_period))
        if isinstance(input_period, Real) else input_period
    )


PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION = {
    Frequency.INTEGER: IntegerPeriod,
    Frequency.YEARLY: YearlyPeriod,
    Frequency.HALFYEARLY: HalfyearlyPeriod,
    Frequency.QUARTERLY: QuarterlyPeriod,
    Frequency.MONTHLY: MonthlyPeriod,
    Frequency.DAILY: DailyPeriod,
    Frequency.UNKNOWN: UnknownPeriod,
}


def periods_from_sdmx_strings(
    sdmx_strings: Iterable[str],
    frequency: Frequency | None = None,
) -> tuple[Period]:
    """
    """
    sdmx_strings = tuple(sdmx_strings)
    if not sdmx_strings:
        return ()
    frequency = (
        Frequency.from_sdmx_string(sdmx_strings[0], )
        if frequency is None else frequency
    )
    period_class = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[frequency]
    return tuple(period_class.from_sdmx_string(i) for i in sdmx_strings)


def periods_from_iso_strings(
    iso_strings: Iterable[str],
    *,
    frequency: Frequency | None = None,
) -> tuple[Period]:
    """
    """
    if frequency is None:
        frequency = Frequency.DAILY
    period_class = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[frequency]
    return tuple(period_class.from_iso_string(i, ) for i in iso_strings)


def periods_from_python_dates(
    python_dates: Iterable[_dt.date],
    *,
    frequency: Frequency | None = None,
) -> tuple[Period]:
    """
    """
    if frequency is None:
        frequency = Frequency.DAILY
    return tuple(Period.from_python_date(i, frequency=frequency, ) for i in python_dates)


def get_encompassing_span(
    *args: tuple[ResolutionContextProtocol | None, ...],
) -> tuple[Span, Period, Period, ]:
    """
    """
    start_dates = tuple(_get_period(i, "start_date", min, ) for i in args if i is not None)
    end_periods = tuple( _get_period(i, "end_date", max, ) for i in args if i is not None)
    start_dates = tuple(i for i in start_dates if i is not None)
    end_periods = tuple(i for i in end_periods if i is not None)
    start = min(start_dates) if start_dates else None
    end = max(end_periods) if end_periods else None
    return Span(start, end), start, end,


def _get_period(something, attr_name, select_func, ) -> Period | None:
    if hasattr(something, attr_name, ):
        return getattr(something, attr_name, )
    try:
        return select_func(i for i in something if i is not None)
    except:
        return None


def periods_from_until(
    start_per: Period,
    end_per: Period,
    step: int = 1,
) -> tuple[Period]:
    """
    """
    _check_periods(start_per, end_per, )
    serials = range(start_per.serial, end_per.serial + 1, step, )
    period_class = type(start_per)
    return tuple(period_class(x) for x in serials)


periods_from_to = periods_from_until


def ensure_period_tuple(
    period_or_string: Iterable[Period] | str,
    frequency: Frequency | None = None,
) -> tuple[Period, ...]:
    """
    """
    #[
    if isinstance(period_or_string, str):
        return _period_tuple_from_string(period_or_string, frequency, )
    else:
        return tuple(period_or_string)
    #]


def _period_tuple_from_string(
    period_string: str,
    frequency: Frequency | None = None,
) -> tuple[Period, ...]:
    """
    """
    #[
    if "..." in period_string:
        start, end = period_string.split("...")
        return tuple(periods_from_to(*periods_from_sdmx_strings(frequency, (start, end, ))))
    if ">>" in period_string:
        start, end = period_string.split(">>")
        return tuple(periods_from_to(*periods_from_sdmx_strings(frequency, (start, end, ))))
    if "," in period_string:
        return tuple(periods_from_sdmx_strings(frequency, period_string.split(",")))
    return (Period.from_sdmx_string(frequency, period_string), )
    #]


def _is_period(x: Any, ) -> bool:
    return isinstance(x, Period, )


def refrequent(
    period: Period,
    new_freq: Frequency,
    *args,
    **kwargs,
) -> Period:
    """
    Convert period to a new frequency
    """
    year, month, day = period.to_ymd(*args, **kwargs, )
    new_class = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
    return new_class.from_ymd(year, month, day, )


def spans_from_short_span(
    short_span: Iterable[Period],
    max_lag: int = 0,
    max_lead: int = 0,
) -> Span:
    r"""
    """
    short_span = tuple(short_span)
    short_span = periods_from_until(short_span[0], short_span[-1], )
    long_span = periods_from_until(short_span[0]+max_lag, short_span[-1]+max_lead, )
    return short_span, long_span,


def spans_from_long_span(
    long_span: Iterable[Period],
    max_lag: int = 0,
    max_lead: int = 0,
) -> Span:
    r"""
    """
    long_span = tuple(long_span)
    long_span = periods_from_until(long_span[0], long_span[-1], )
    short_span = periods_from_until(long_span[0]-max_lag, long_span[-1]-max_lead, )
    return short_span, long_span,


convert_to_new_freq = refrequent


SPAN_ELLIPSIS = "…"


def get_printable_span(start: Period, end: Period, ) -> str:
    """
    """
    return (
        f"{start}{SPAN_ELLIPSIS}{end}"
        if start is not None or end is not None
        else "None"
    )


#
# Legacy aliases
#


class Dater(Period, ):

    @staticmethod
    def from_sdmx_string(frequency, sdmx_string):
        return Period.from_sdmx_string(sdmx_string, frequency=frequency, )

    @staticmethod
    def from_iso_string(frequency, iso_string):
        return Period.from_iso_string(iso_string, frequency=frequency, )


class Ranger(Span, ):
    pass


EmptyRanger = EmptySpan


daters_from_to = periods_from_to


def daters_from_iso_strings(frequency, iso_strings):
    return periods_from_iso_strings(iso_strings, frequency=frequency, )


def daters_from_sdmx_strings(frequency, sdmx_strings):
    return periods_from_sdmx_strings(sdmx_strings, frequency=frequency, )


DATER_CLASS_FROM_FREQUENCY_RESOLUTION = PERIOD_CLASS_FROM_FREQUENCY_RESOLUTION

