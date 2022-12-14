
from __future__ import annotations

from datetime import date
from typing import Union, Optional, Self, Callable, Iterable
from numbers import Number
from enum import Flag
from collections.abc import Sequence


class Freq(Flag):
    #(
    INTEGER = 0
    YEARLY = 1
    HALFYEARLY = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365
    REGULAR = YEARLY | HALFYEARLY | QUARTERLY | MONTHLY

    @property
    def letter(self) -> str:
        return FREQ_LETTERS[self]

    @property
    def sdmx_format(self) -> str:
        return FREQ_SDMX_FORMATS[self]
    #)


FREQ_LETTERS = {
    Freq.INTEGER: 'I',
    Freq.YEARLY: 'Y',
    Freq.HALFYEARLY: 'H',
    Freq.QUARTERLY: 'Q',
    Freq.MONTHLY: 'M',
    Freq.WEEKLY: 'W',
    Freq.DAILY: 'D',
}


FREQ_SDMX_FORMATS = {
    Freq.INTEGER: '({serial})',
    Freq.YEARLY: '{year:04g}',
    Freq.HALFYEARLY: '{year:04g}-{letter}{per:1g}',
    Freq.QUARTERLY: '{year:04g}-{letter}{per:1g}',
    Freq.MONTHLY: '{year:04g}-{per:02g}',
    Freq.WEEKLY: '{year:04g}-{per:02g}',
    Freq.DAILY: '{year:04g}-{month:02g}-{day:02g}',
}


def _check_daters(first, second) -> None:
    if type(first) is not type(second):
        raise Exception("Input dates must be the same frequency")


def _check_daters_decorate(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        _check_daters(args[0], args[1])
        return func(*args, **kwargs)
    return wrapper


def _check_offset(offset) -> None:
    if not isinstance(offset, int):
        raise Exception("Date offset must be an integer")


def _check_offset_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        _check_offset(args[1])
        return func(*args, **kwargs)
    return wrapper


def _remove_blanks_decorate(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).replace(" ","")
    return wrapper


class Dater:
    #(
    freq = None


    def __init__(self, serial=0):
        self.serial = int(serial)


    def __repr__(self) -> str:
        return self.__str__()


    def __iter__(self) -> Iterable:
        yield self


    @_check_offset_decorator
    def __add__(self, other: int) -> Self:
        return type(self)(self.serial + int(other))


    __radd__ = __add__


    def __sub__(self, other: Union[Self, int]) -> Union[Self, int]:
        if isinstance(other, Dater):
            return self.diff(other)
        else:
            return self.__add__(-int(other))


    @_check_daters_decorate
    def diff(self, other: Self) -> int:
        return self.serial - other.serial


    @_check_daters_decorate
    def __rshift__(self, end_date: Self) -> Ranger:
        return Ranger(self, end_date, 1) 


    @_check_daters_decorate
    def __lshift__(self, end_date: Self) -> Ranger:
        return Ranger(start_date, end_date, -1) 


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
    #)


class IntegerDater(Dater):
    #(
    freq = Freq.INTEGER


    def __repr__(self) -> str:
        return f"ii({self.serial})"


    def __str__(self) -> str:
        serial = self.serial
        return self.freq.sdmx_format.format(serial=serial)
    #)


class DailyDater(Dater):
    #(
    freq: Freq = Freq.DAILY


    @_remove_blanks_decorate
    def __repr__(self) -> str:
        return f"dd{self.year_month_day()}"


    @classmethod
    def serial_from_ymd(cls: type, year: int, month: int=1, day: int=1) -> int:
        return date(year, month, day).toordinal()


    @classmethod
    def from_ymd(cls: type, *args):
        return cls(cls.serial_from_ymd(*args))


    @classmethod
    def ymd_from_serial(cls: type, serial: int) -> tuple[int, int, int]:
        temp = date.fromordinal(serial)
        return temp.year, temp.month, temp.day


    def yp_from_serial(self) -> tuple[int, int]:
        boy_serial = date(date.fromordinal(self.serial).year, 1, 1)
        per = self.serial - boy_serial + 1
        year = date.fromordinal(self.serial).year
        return year, per


    @property
    def year_month_day(self) -> tuple[int, int, int]:
        return self.ymd_from_serial(self.serial)


    @property
    def year_per(self) -> tuple[int, int]:
        return self.yp_from_serial()


    def __str__(self) -> str:
        year, month, day = self.year_month_day
        letter = self.freq.letter
        return self.freq.sdmx_format.format(year=year, month=month, day=day, letter=letter)
    #)


class RegularDater:
    #(
    @classmethod
    def serial_from_yp(cls: type, year: int, per: int) -> int:
        return int(year)*int(cls.freq.value) + int(per) - 1


    def yp_from_serial(self) -> tuple[int, int]:
        return self.serial//self.freq.value, self.serial%self.freq.value+1


    @classmethod
    def from_yp(cls: type, year: int, per: int=1) -> Self:
        new_serial = cls.serial_from_yp(year, per)
        return cls(new_serial)


    @property
    def year_period(self) -> tuple[int, int]:
        return self.yp_from_serial()


    @property
    def year(self) -> int:
        return self.yp_from_serial()[0]


    def __str__(self) -> str:
        year, per = self.year_period
        letter = self.freq.letter
        return self.freq.sdmx_format.format(year=year, per=per, letter=letter)
    #)


class YearlyDater(Dater, RegularDater): 
    freq: Freq = Freq.YEARLY
    @_remove_blanks_decorate
    def __repr__(self) -> str: return f"yy{self.year}"


class HalfyearlyDater(Dater, RegularDater):
    freq: Freq = Freq.HALFYEARLY
    @_remove_blanks_decorate
    def __repr__(self) -> str: return f"hh{self.year_period}"


class QuarterlyDater(Dater, RegularDater):
    freq: Freq = Freq.QUARTERLY
    @_remove_blanks_decorate
    def __repr__(self) -> str: return f"qq{self.year_period}"


class MonthlyDater(Dater, RegularDater):
    freq: Freq = Freq.MONTHLY
    @_remove_blanks_decorate
    def __repr__(self) -> str: return f"mm{self.year_period}"


yy = YearlyDater.from_yp
hh = HalfyearlyDater.from_yp
qq = QuarterlyDater.from_yp
mm = MonthlyDater.from_yp
dd = DailyDater.from_ymd
ii = IntegerDater


class Ranger():
    #(
    def __init__(self, start_date: Dater, end_date: Dater, step: int=1) -> None:
        _check_daters(start_date, end_date)
        start_serial = start_date.serial
        end_serial = end_date.serial
        self._class = type(start_date)
        self._serials = range(start_serial, end_serial+_sign(step), step)


    @property
    def freq(self) -> Freq:
        return self._class.freq


    @property
    def start_date(self) -> Dater:
        return self._class(self._serials.start)


    @property
    def end_date(self) -> Dater:
        return self._class(self._serials.stop-_sign(self.step))


    @property
    def step(self) -> int:
        return self._serials.step


    def __len__(self) -> int:
        return len(self._serials)


    def __getitem__(self, key: Union[int, slice]) -> Union[Self, Dater]:
        if isinstance(key, slice):
            self._serials = self._serials[key]
            return self
        else:
            return self._class(self._serials[key])


    def __str__(self) -> str:
        step_string = f", {self.step}" if self.step!=1 else ""
        start_date_str = self.start_date.__str__()
        end_date_str = self.end_date.__str__()
        return f"Ranger({start_date_str}, {end_date_str}{step_str})"


    def __repr__(self) -> str:
        step_rep = f", {self.step}" if self.step!=1 else ""
        start_date_rep = self.start_date.__repr__()
        end_date_rep = self.end_date.__repr__()
        return f"Ranger({start_date_rep}, {end_date_rep}{step_rep})"


    def __add__(self, offset: int) -> range:
        return Ranger(self.start_date+offset, self.end_date+offset, self.step)


    __radd__ = __add__


    def __sub__(self, offset: Union[Dater, int]) -> Union[range, Self]:
        if isinstance(offset, Dater):
            return range(self.start_date-offset, self.end_date-offset, self.step)
        else:
            return Ranger(self.start_date-offset, self.end_date-offset, self.step)


    def __rsub__(self, other: Dater) -> range:
        if isinstance(other, Dater):
            return range(other-self.start_date, other-self.end_date, -self.step)


    def __iter__(self) -> Iterable:
        return (self._class(x) for x in self._serials)
    #)


def _sign(x: Number) -> int:
    return 1 if x>0 else (0 if x==0 else -1)


def date_index(dates: Iterable[Dater], base: Dater) -> Iterable[int]:
    return (x-base for x in dates)


