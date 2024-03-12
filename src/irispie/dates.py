"""
Dates, date ranges, and date frequencies
"""


#[
from __future__ import annotations

import re as _re
import datetime as _dt
import enum as _en
import functools as _ft
from typing import (Union, Self, Any, Protocol, TypeAlias, runtime_checkable, )
from collections.abc import (Iterable, Callable, Iterator, )
from numbers import (Real, )

from . import pages as _pages
from .conveniences import copies as _copies
from . import wrongdoings as _wrongdoings
#]


__all__ = [
    "Frequency", "Freq",
    "yy", "hh", "qq", "mm", "dd", "ii",
    "Ranger", "EmptyRanger", "start", "end",
    "Dater", "daters_from_sdmx_strings", "daters_from_iso_strings", "daters_from_to",
    "YEARLY", "HALFYEARLY", "QUARTERLY", "MONTHLY", "WEEKLY", "DAILY",
    "DATER_CLASS_FROM_FREQUENCY_RESOLUTION",
    "convert_to_new_freq",
]


class Frequency(_en.IntEnum):
    """
    Enumeration of date frequencies
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
    def from_letter(
        klass,
        string: str,
        /,
    ) -> Self:
        """
        """
        letter = string.replace("_", "").upper()[0]
        return next( x for x in klass if x.name.startswith(letter) )

    @classmethod
    def from_sdmx_string(
        klass,
        sdmx_string: str,
        /,
    ) -> Self:
        """
        """
        try:
            return next(
                freq for freq, (length, pattern, ) in SDMX_REXP_FORMATS.items()
                if (length is None or len(sdmx_string) == length) and pattern.fullmatch(sdmx_string, )
            )
        except StopIteration:
            raise _wrongdoings.IrisPieCritical(
                f"Cannot determine date frequency from \"{sdmx_string}\"; "
                f"probably not a valid SDMX string"
            )

    @property
    def letter(self, /, ) -> str:
        return self.name[0] if self is not self.UNKNOWN else "?"

    @property
    def plotly_format(self, /, ) -> str:
        return PLOTLY_FORMATS[self]

    @property
    def is_regular(self, /, ) -> bool:
        return self in (self.YEARLY, self.HALFYEARLY, self.QUARTERLY, self.MONTHLY, )

    def __str__(self, /, ) -> str:
        return self.name

    #]


Freq = Frequency


YEARLY = Frequency.YEARLY
HALFYEARLY = Frequency.HALFYEARLY
QUARTERLY = Frequency.QUARTERLY
MONTHLY = Frequency.MONTHLY
WEEKLY = Frequency.WEEKLY
DAILY = Frequency.DAILY


PLOTLY_FORMATS = {
    Frequency.YEARLY: "%Y",
    Frequency.HALFYEARLY: "%Y-%m",
    Frequency.QUARTERLY: "%Y-Q%q",
    Frequency.MONTHLY: "%Y-%m",
    Frequency.WEEKLY: "%Y-%W",
    Frequency.DAILY: "%Y-%m-%d",
    Frequency.INTEGER: None,
}


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
    Context protocol for contextual date resolution
    """
    start_date = ...
    end_date = ...


class ResolutionContext:
    """
    """
    #[

    def __init__(
        self,
        start_date: _dates.Dater | None = None,
        end_date: _dates.Dater | None = None,
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
    Contextual date protocol
    """
    needs_resolve = ...
    def resolve(self, context: ResolutionContextProtocol) -> Any: ...


def _check_daters(first, second, ) -> None:
    if str(type(first)) == str(type(second)):
        return
    message = "Cannot handle dates of different types in this context"
    raise _wrongdoings.IrisPieError(message, )


def _check_daters_decorate(func: Callable, ) -> Callable:
    def wrapper(*args, **kwargs):
        _check_daters(args[0], args[1], )
        return func(*args, **kwargs, )
    return wrapper


def _check_offset(offset, ) -> None:
    if not isinstance(offset, int, ):
        message = "Date offset must be an integer"
        raise Exception(message, )


def _check_offset_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        _check_offset(args[1])
        return func(*args, **kwargs)
    return wrapper


def _remove_blanks(func: Callable,) -> Callable:
    def wrapper(*args, **kwargs, ):
        return func(*args, **kwargs, ).replace(" ", "", )
    return wrapper


class RangeableMixin:
    #[
    def __rshift__(self, end_date: Self | None, ) -> Ranger:
        """
        Dater >> Dater or Dater >> None
        """
        return Ranger(self, end_date, 1)

    def __rrshift__(self, start_date: Self | None, ) -> Ranger:
        """
        None >> Dater
        """
        return Ranger(start_date, self, 1)

    def __lshift__(self, start_date: Self | None, ) -> Ranger:
        """
        Dater << Dater or Dater << None
        """
        return Ranger(start_date, self, -1) 

    def __rlshift__(self, end_date: Self | None, ) -> Ranger:
        """
        None << Dater
        """
        return Ranger(self, end_date, -1) 
    #]


@_pages.reference(
    path=("data_management", "dates.md", ),
    categories={
        "constructor": "Creating new dates",
        "property": None,
    },
)
class Dater(
    RangeableMixin,
    _copies.CopyMixin,
):
    """
......................................................................

Dates and date ranges
======================

......................................................................
    """
    #[
    frequency = None
    needs_resolve = False

    @staticmethod
    def from_iso_string(freq: Frequency, iso_string: str, ) -> Dater:
        """
        """
        year, month, day = iso_string.split("-", )
        return DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_ymd(int(year), int(month), int(day), )

    @staticmethod
    def from_sdmx_string(freq: Frequency, sdmx_string: str, ) -> Dater:
        """
        """
        freq = Frequency.from_sdmx_string(sdmx_string, ) if freq is None else freq
        return DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_sdmx_string(sdmx_string)

    @staticmethod
    def from_ymd(freq: Frequency, *args, ) -> Dater:
        """
        """
        return DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_ymd(*args, )

    dater_from_ymd = from_ymd

    @staticmethod
    def today(freq: Frequency, ) -> Dater:
        """
        """
        t = _dt.date.today()
        return DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq].from_ymd(t.year, t.month, t.day, )

    @property
    def start_date(self, /, ) -> Self:
        """
        """
        return self

    @property
    def end_date(self, /, ) -> Self:
        """
        """
        return self

    def convert_to_new_freq(self, new_freq: Frequency, *args ,**kwargs, ) -> Dater:
        """
        """
        year, month, day = self.to_ymd(*args, **kwargs, )
        new_class = DATER_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
        return new_class.from_ymd(year, month, day, )

    convert = convert_to_new_freq

    def to_iso_string(
        self,
        /,
        position: Literal["start", "middle", "end", ] = "start",
    ) -> str:
        year, month, day = self.to_ymd(position=position, )
        return f"{year:04g}-{month:02g}-{day:02g}"

    def to_python_date(
        self,
        /,
        position: Literal["start", "middle", "end", ] = "start",
    ) -> str:
        return _dt.date(*self.to_ymd(position=position, ))

    to_plotly_date = _ft.partialmethod(to_iso_string, position="middle", )

    def __init__(self, serial=0, ) -> None:
        self.serial = int(serial)

    def get_distance_from_origin(self) -> int:
        return self.serial - self.origin

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
        return ("{date_str:"+str_format+"}").format(date_str=self.__str__())

    def __iter__(self) -> Iterator[Self]:
        yield self

    def __hash__(self, /, ) -> int:
        return hash((int(self.serial), hash(self.frequency), ))

    @_check_offset_decorator
    def __add__(self, other: int) -> Self:
        return type(self)(self.serial + int(other))

    __radd__ = __add__

    def __sub__(self, other: Self | int) -> Self | int:
        if _is_dater(other, ):
            return self._sub_dater(other)
        else:
            return self.__add__(-int(other))

    @_check_daters_decorate
    def _sub_dater(self, other: Self) -> int:
        return self.serial - other.serial

    def __index__(self):
        return self.serial

    @_check_daters_decorate
    def __eq__(self, other) -> bool:
        return self.serial == other.serial

    @_check_daters_decorate
    def __ne__(self, other) -> bool:
        return self.serial != other.serial

    @_check_daters_decorate
    def __lt__(self, other) -> bool:
        return self.serial < other.serial

    @_check_daters_decorate
    def __le__(self, other) -> bool:
        return self.serial <= other.serial

    @_check_daters_decorate
    def __gt__(self, other) -> bool:
        return self.serial > other.serial

    @_check_daters_decorate
    def __ge__(self, other) -> bool:
        return self.serial >= other.serial

    def shift(
        self,
        by: int | str = -1,
    ) -> None:
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


class IntegerDater(Dater, ):
    """
    """
    #[

    frequency = Frequency.INTEGER
    needs_resolve = False
    origin = 0
    _PLOTLY_DATE_FACTORY = {
        "start": lambda x: x - 0.5,
        "middle": lambda x: x,
        "end": lambda x: x + 0.5,
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> IntegerDater:
        sdmx_string = sdmx_string.strip().removeprefix("(").removesuffix(")")
        return klass(int(sdmx_string))

    def to_sdmx_string(self) -> str:
        return f"({self.serial})"

    def __repr__(self) -> str:
        return f"ii({self.serial})"

    def to_plotly_date(
        self,
        /,
        position: Literal["start", "middle", "end", ] = "middle",
    ) -> Real:
        return self._PLOTLY_DATE_FACTORY[position](self.serial, )

    #]


class DailyDater(Dater, ):
    """
    """
    #[

    frequency: Frequency = Frequency.DAILY
    needs_resolve = False
    origin = _dt.date(BASE_YEAR, 1, 1).toordinal()

    @classmethod
    def from_ymd(klass: type, year: int, month: int=1, day: int=1) -> Self:
        serial = _dt.date(year, month, day).toordinal()
        return klass(serial)

    @classmethod
    def from_year_period(klass, year: int, period: int) -> Self:
        boy_serial = _dt.date(year, 1, 1).toordinal()
        serial = boy_serial + int(period) - 1
        return klass(serial)

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> DailyDater:
        year, month, day, *_ = sdmx_string.split("-")
        return klass.from_ymd(int(year), int(month), int(day))

    @classmethod
    def from_iso_string(klass, iso_string: str, ) -> Dater:
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
    def period(self, /, ) -> int:
        return self.to_year_period()[1]

    def to_sdmx_string(self, /, **kwargs) -> str:
        year, month, day = self.to_ymd()
        return f"{year:04g}-{month:02g}-{day:02g}"

    def to_year_period(self) -> tuple[int, int]:
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
        year, per = self.to_year_period()
        return self.from_year_period(year, 1) if per != 1 else None

    def to_daily(self, /, **kwargs, ) -> Self:
        return self

    #]


def _serial_from_ypf(year: int, per: int, freq: int) -> int:
    return int(year)*int(freq) + int(per) - 1


class RegularDaterMixin:
    """
    """
    #[

    @classmethod
    def from_year_period(
            klass: type,
            year: int,
            per: int | str = 1,
        ) -> Self:
        per = per if per != "end" else klass.frequency.value
        new_serial = _serial_from_ypf(year, per, klass.frequency.value)
        return klass(new_serial)

    @classmethod
    def from_ymd(klass, year: int, month: int=1, day: int=1, ) -> YearlyDater:
        return klass.from_year_period(year, klass.month_to_period(month, ), )

    @classmethod
    def from_iso_string(klass, iso_string: str, ) -> Dater:
        """
        """
        year, month, day, *_ = iso_string.split("-", )
        return klass.from_ymd(int(year), int(month), int(day), )

    @property
    def year(self, ) -> int:
        return self.to_year_period()[0]

    @property
    def period(self, ) -> int:
        return self.to_year_period()[1]

    def to_year_period(self) -> tuple[int, int]:
        return self.serial//self.frequency.value, self.serial%self.frequency.value+1

    def get_year(self) -> int:
        return self.to_year_period()[0]

    def to_ymd(
        self, 
        /,
        position: Literal["start", "middle", "end", ] = "middle",
    ) -> tuple[int, int, int]:
        year, per = self.to_year_period()
        return year, *self.month_day_resolution[position][per]

    def __str__(self, /, ) -> str:
        return self.to_sdmx_string()

    def create_soy(self, ) -> Self:
        year, *_ = self.to_year_period()
        return self.from_year_period(year, 1)

    create_boy = create_soy

    def create_eoy(self, ) -> Self:
        year, *_ = self.to_year_period()
        return self.from_year_period(year, "end")

    def create_eopy(self, ) -> Self:
        year, *_ = self.to_year_period()
        return self.from_year_period(year-1, self.frequency.value)

    def create_tty(self, ) -> Self:
        year, per = self.to_year_period()
        return self.from_year_period(year, 1) if per != 1 else None

    def to_daily(
        self,
        /,
        position: Literal["start", "middle", "end", ] = "middle"
    ) -> DailyDater:
        try:
            return DailyDater.from_ymd(*self.to_ymd(position=position, ), )
        except:
            raise IrisPieCritical("Cannot convert date to daily date.")

    #]


class YearlyDater(RegularDaterMixin, Dater, ):
    """
    """
    #[

    frequency: Frequency = Frequency.YEARLY
    needs_resolve: bool = False
    origin = _serial_from_ypf(BASE_YEAR, 1, Frequency.YEARLY)
    month_day_resolution = {
        "start": {1: (1, 1)},
        "middle": {1: (6, 30)},
        "end": {1: (12, 31)},
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> YearlyDater:
        return klass(int(sdmx_string))

    def to_sdmx_string(self) -> str:
        return f"{self.get_year():04g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"yy({self.get_year()})"

    @staticmethod
    def month_to_period(month: int, ) -> int:
        return 1

    #]


class HalfyearlyDater(RegularDaterMixin, Dater, ):
    #[
    frequency: Frequency = Frequency.HALFYEARLY
    needs_resolve: bool = False
    origin = _serial_from_ypf(BASE_YEAR, 1, Frequency.HALFYEARLY)
    month_day_resolution = {
        "start": {1: (1, 1), 2: (7, 1)},
        "middle": {1: (3, 15), 2: (9, 15)},
        "end": {1: (6, 30), 2: (12, 31)}
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> HalfyearlyDater:
        year, halfyear = sdmx_string.split("-H")
        return klass.from_year_period(int(year), int(halfyear))

    def to_sdmx_string(self) -> str:
        year, per = self.to_year_period()
        return f"{year:04g}-{self.frequency.letter}{per:1g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"hh{self.to_year_period()}"

    def to_ymd(
        self, 
        /,
        position: Literal["start", "middle", "end", ] = "middle",
    ) -> tuple[int, int, int]:
        year, per = self.to_year_period()
        return (
            year,
            *{"start": (1, 1), "middle": (6, 3), "end": (12, 31)}[position],
        )

    def get_month(
        self,
        /,
        position: Literal["start", "middle", "end", ] = "middle",
    ) -> int:
        _, per = self.to_year_period()
        return month_resolution[position][per]

    @staticmethod
    def month_to_period(month: int, ) -> int:
        return 1+((month-1)//6)
    #]


class QuarterlyDater(RegularDaterMixin, Dater, ):
    """
    """
    #[

    frequency: Frequency = Frequency.QUARTERLY
    needs_resolve: bool = False
    origin = _serial_from_ypf(BASE_YEAR, 1, Frequency.QUARTERLY)
    month_day_resolution = {
        "start": {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)},
        "middle": {1: (2, 15), 2: (5, 15), 3: (8, 15), 4: (11, 15)},
        "end": {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)},
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> QuarterlyDater:
        year, quarter = sdmx_string.split("-Q")
        return klass.from_year_period(int(year), int(quarter))

    def to_sdmx_string(self) -> str:
        year, per = self.to_year_period()
        return f"{year:04g}-{self.frequency.letter}{per:1g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"qq{self.to_year_period()}"

    @staticmethod
    def month_to_period(month: int, ) -> int:
        return 1+((month-1)//3)

    #]


class MonthlyDater(RegularDaterMixin, Dater, ):
    #[
    frequency: Frequency = Frequency.MONTHLY
    needs_resolve: bool = False
    origin = _serial_from_ypf(BASE_YEAR, 1, Frequency.MONTHLY)
    month_day_resolution = {
        "start": {1: (1, 1), 2: (2, 1), 3: (2, 1), 4: (4, 1), 5: (5, 1), 6: (6, 1), 7: (7, 1), 8: (8, 1), 9: (9, 1), 10: (10, 1), 11: (11, 1), 12: (12, 1)},
        "middle": {1: (1, 15), 2: (2, 15), 3: (2, 15), 4: (4, 15), 5: (5, 15), 6: (6, 15), 7: (7, 15), 8: (8, 15), 9: (9, 15), 10: (10, 15), 11: (11, 15), 12: (12, 15)},
        "end": {1: (1, 31), 2: (2, 28), 3: (3, 31), 4: (4, 30), 5: (5, 31), 6: (6, 30), 7: (7, 31), 8: (8, 31), 9: (9, 30), 10: (10, 31), 11: (11, 30), 12: (12, 31)},
    }

    @classmethod
    def from_sdmx_string(klass, sdmx_string: str) -> MonthlyDater:
        year, month = sdmx_string.split("-")
        return klass.from_year_period(int(year), int(month))

    def to_sdmx_string(self) -> str:
        year, per = self.to_year_period()
        return f"{year:04g}-{per:02g}"

    @_remove_blanks
    def __repr__(self) -> str: return f"mm{self.to_year_period()}"

    @staticmethod
    def month_to_period(month: int, ) -> int:
        return month
    #]


class UnknownDater:
    """
    """
    #[

    def from_sdmx_string(sdmx_string: str) -> None:
        return None

    #]


def _dater_or_ranger_decorator(
    func: Callable,
    /,
) -> Callable:
    """
    """
    #[
    def wrapper(*args, ):
        try:
            index = args.index(Ellipsis)
            start_date = func(*args[:index], ) if args[:index] else None
            end_date = func(*args[index+1:], ) if args[index+1:] else None
            return Ranger(start_date, end_date, )
        except ValueError:
            return func(*args, )
    return wrapper
    #]


yy = _dater_or_ranger_decorator(YearlyDater.from_year_period, )
yy.__doc__ = \
"""
------------------------------------------------------------


`yy`
=====

##### Create a yearly-frequency date or date range ####

Syntax
-------

    date = yy(year)
    range = yy(start_year, ..., end_year)

Input arguments
----------------

### `year ###
Year (an integer number).

### `start_year` ###
Start year for a date range.

### `end_year` ###
End year for a date range.

Returns
--------

### `date` ###
A `YearlyDater` object (if only a single year is specified as an input
argument).

### `range` ###
A `Ranger` object (if a start and end year are specified as input
arguments).


------------------------------------------------------------
"""


hh = _dater_or_ranger_decorator(HalfyearlyDater.from_year_period, )
hh.__doc__ = \
"""
------------------------------------------------------------


`hh`
=====


------------------------------------------------------------
"""


qq = _dater_or_ranger_decorator(QuarterlyDater.from_year_period)
mm = _dater_or_ranger_decorator(MonthlyDater.from_year_period)
ii = _dater_or_ranger_decorator(IntegerDater)


def dd(year: int, month: int | ellipsis, day: int) -> DailyDater:
    if month is Ellipsis:
        return DailyDater.from_year_period(year, day)
    else:
        return DailyDater.from_ymd(year, month, day)


class Ranger(_copies.CopyMixin, ):
    """
    """
    #[

    def __init__(
        self, 
        from_date: Dater | None = None,
        until_date: Dater | None = None,
        step: int = 1,
    ) -> None:
        """
        Date range constructor
        """
        from_date = resolve_dater_or_integer(from_date)
        until_date = resolve_dater_or_integer(until_date)
        if step > 0:
            default_from_date = start
            default_until_date = end
        else:
            default_from_date = end
            default_until_date = start
        self._start_date = from_date if from_date is not None else default_from_date
        self._end_date = until_date if until_date is not None else default_until_date
        self._step = step
        self.needs_resolve = self._start_date.needs_resolve or self._end_date.needs_resolve
        if not self.needs_resolve:
            _check_daters(from_date, until_date)

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def step(self):
        return self._step

    @property
    def _class(self):
        return type(self._start_date) if not self.needs_resolve else None

    @property
    def direction(self, ) -> Literal["forward", "backward", ]:
        return "forward" if self._step > 0 else "backward"

    @property
    def _serials(self) -> range | None:
        return range(self._start_date.serial, self._end_date.serial+_sign(self._step), self._step) if not self.needs_resolve else None

    # @property
    # def needs_resolve(self) -> bool:
        # return bool(self._start_date and self._end_date)

    @property
    def frequency(self) -> Frequency:
        return self._class.frequency

    def reverse(self, ) -> None:
        """
        """
        #[
        self._start_date, self._end_date = self._end_date, self._start_date
        self._step = -self._step
        #]

    def reversed(self, ) -> Ranger:
        """
        """
        #[
        new = self.copy()
        new.reverse()
        return new
        #]

    def shift_end_date(
        self,
        k: int,
        /,
    ) -> Self:
        """
        """
        return Ranger(self._start_date, self._end_date+k, self._step, )

    def shift_start_date(
        self,
        k: int,
        /,
    ) -> Self:
        """
        """
        return Ranger(self._start_date+k, self._end_date, self._step, )

    def to_plotly_dates(self, *args, **kwargs, ) -> Iterable[str]:
        return [t.to_plotly_date(*args, **kwargs, ) for t in self]

    def to_iso_strings(self, *args, **kwargs, ) -> Iterable[str]:
        return [t.to_iso_string(*args, **kwargs, ) for t in self]

    def to_python_dates(self, *args, **kwargs, ) -> Iterable[str]:
        return [t.to_python_date(*args, **kwargs, ) for t in self]

    def __len__(self) -> int|None:
        return len(self._serials) if not self.needs_resolve else None

    def __str__(self) -> str:
        step_str = f", {self._step}" if self._step!=1 else ""
        start_date_str = self._start_date.__str__()
        end_date_str = self._end_date.__str__()
        return f"Ranger({start_date_str}, {end_date_str}{step_str})"

    def __repr__(self) -> str:
        step_rep = f", {self._step}" if self._step!=1 else ""
        start_date_rep = self._start_date.__repr__()
        end_date_rep = self._end_date.__repr__()
        return f"Ranger({start_date_rep}, {end_date_rep}{step_rep})"

    def __add__(self, offset: int) -> range:
        return Ranger(self._start_date+offset, self._end_date+offset, self._step)

    __radd__ = __add__

    def __sub__(self, offset: Dater | int) -> range | Self:
        if _is_dater(offset, ):
            return range(self._start_date-offset, self._end_date-offset, self._step) if not self.needs_resolve else None
        else:
            return Ranger(self._start_date-offset, self._end_date-offset, self._step)

    def __rsub__(self, ather: Dater) -> range|None:
        if _is_dater(other, ):
            return range(other-self._start_date, other-self._end_date, -self._step) if not self.needs_resolve else None
        else:
            return None

    def __iter__(self) -> Iterable:
        return (self._class(x) for x in self._serials) if not self.needs_resolve else None

    def __getitem__(self, i: int) -> Dater|None:
        return self._class(self._serials[i]) if not self.needs_resolve else None

    def resolve(self, context: ResolutionContextProtocol, /, ) -> Self:
        resolved_start_date = self._start_date if self._start_date else self._start_date.resolve(context, )
        resolved_end_date = self._end_date if self._end_date else self._end_date.resolve(context, )
        return Ranger(resolved_start_date, resolved_end_date, self._step, )

    def shift(self, by: int) -> None:
        self._start_date += by
        self._end_date += by

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __eq__(self, other: Self) -> bool:
        return self._start_date==other._start_date and self._end_date==other._end_date and self._step==other._step

    def __bool__(self) -> bool:
        return not self.needs_resolve

    #]


class EmptyRanger:
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


def date_index(dates: Iterable[Dater | None], base: Dater) -> Iterable[int]:
    """
    """
    return (
        (t - base) if t is not None else None
        for t in dates
    )


class ContextualDater(Dater, RangeableMixin, ):
    """
    Dates with context dependent resolution
    """
    #[
    needs_resolve = True

    def __init__(self, resolve_from: str, offset: int=0) -> None:
        self._resolve_from = resolve_from
        self._offset = offset

    def __add__(self, offset: int) -> None:
        return type(self)(self._resolve_from, self._offset+offset)

    def __sub__(self, offset: int) -> None:
        return type(self)(self._resolve_from, self._offset-offset)

    def __str__(self) -> str:
        return "<>." + self._resolve_from + (f"{self._offset:+g}" if self._offset else "")

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return False

    def resolve(self, context: ResolutionContextProtocol) -> Dater:
        return getattr(context, self._resolve_from) + self._offset
    #]


start = ContextualDater("start_date")
end = ContextualDater("end_date")


def resolve_dater_or_integer(input_date: Any) -> Dater:
    """
    Convert non-dater to integer dater
    """
    if isinstance(input_date, Real):
        input_date = IntegerDater(int(input_date))
    return input_date


DATER_CLASS_FROM_FREQUENCY_RESOLUTION = {
    Frequency.INTEGER: IntegerDater,
    Frequency.YEARLY: YearlyDater,
    Frequency.HALFYEARLY: HalfyearlyDater,
    Frequency.QUARTERLY: QuarterlyDater,
    Frequency.MONTHLY: MonthlyDater,
    Frequency.DAILY: DailyDater,
    Frequency.UNKNOWN: UnknownDater,
}


def daters_from_sdmx_strings(freq: Frequency, sdmx_strings: Iterable[str], ) -> Iterable[Dater]:
    """
    """
    dater_class = DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq]
    return ( dater_class.from_sdmx_string(i) for i in sdmx_strings )


def daters_from_iso_strings(freq: Frequency, iso_strings: Iterable[str], ) -> Iterable[Dater]:
    """
    """
    dater_class = DATER_CLASS_FROM_FREQUENCY_RESOLUTION[freq]
    return ( dater_class.from_iso_string(x) for x in iso_strings )


def get_encompassing_span(*args: ResolutionContextProtocol, ) -> Ranger:
    """
    """
    start_dates = tuple(_get_date(x, "start_date", min, ) for x in args)
    end_dates = tuple( _get_date(x, "end_date", max, ) for x in args)
    start_dates = tuple(d for d in start_dates if d is not None)
    end_dates = tuple(d for d in end_dates if d is not None)
    start_date = min(start_dates) if start_dates else None
    end_date = max(end_dates) if end_dates else None
    return Ranger(start_date, end_date), start_date, end_date


def _get_date(something, attr_name, select_func, ) -> Dater | None:
    if hasattr(something, attr_name, ):
        return getattr(something, attr_name, )
    try:
        return select_func(i for i in something if i is not None)
    except:
        return None


def daters_from_to(
    start_date: Dater,
    end_date: Dater,
    /,
) -> tuple[Dater, ...]:
    """
    """
    _check_daters(start_date, end_date, )
    serials = range(start_date.serial, end_date.serial+1, )
    dater_class = type(start_date)
    return tuple(dater_class(x) for x in serials)


def ensure_dater(
    dater_or_string: _dates.Dater | str,
    frequency: _dates.Frequency | None = None,
) -> _dates.Dater:
    """
    """
    #[
    return (
        dater_or_string if not isinstance(dater_or_string, str)
        else Dater.from_sdmx_string(frequency, dater_or_string, )
    )
    #]


def _is_dater(x: Any, ) -> bool:
    return isinstance(x, Dater, )


def convert_to_new_freq(self, new_freq: Frequency, *args ,**kwargs, ) -> Dater:
    """
    Convert date to a new frequency
    """
    year, month, day = self.to_ymd(*args, **kwargs, )
    new_class = DATER_CLASS_FROM_FREQUENCY_RESOLUTION[new_freq]
    return new_class.from_ymd(year, month, day, )

