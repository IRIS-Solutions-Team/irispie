
from __future__ import annotations
from datetime import date
from typing import Union, Optional
from enum import Enum

class Freq(Enum):
    INTEGER = 0
    YEARLY = 1
    HALF_YEARLY = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365

    def is_regular(self) -> bool:
        return self in [
            self.YEARLY,
            self.HALF_YEARLY,
            self.QUARTERLY,
            self.MONTHLY,
        ]

    @property
    def letter(self) -> str:
        return FREQ_LETTERS[self]

    @property
    def sdmx_format(self) -> str:
        return FREQ_SDMX_FORMATS[self]


FREQ_LETTERS = {
    Freq.INTEGER: 'I',
    Freq.YEARLY: 'Y',
    Freq.HALF_YEARLY: 'H',
    Freq.QUARTERLY: 'Q',
    Freq.MONTHLY: 'M',
    Freq.WEEKLY: 'W',
    Freq.DAILY: 'D',
}


FREQ_SDMX_FORMATS = {
    Freq.INTEGER: '({serial})',
    Freq.YEARLY: '{year:04g}',
    Freq.HALF_YEARLY: '{year:04g}-{letter}{per:1g}',
    Freq.QUARTERLY: '{year:04g}-{letter}{per:1g}',
    Freq.MONTHLY: '{year:04g}-{per:02g}',
    Freq.WEEKLY: '{year:04g}-{per:02g}',
    Freq.DAILY: '{year:04g}-{month:02g}-{day:02g}',
}


class Dater:
    #(
    _freq = None

    @property
    def freq(self) -> Freq:
        return self._freq

    def __init__(self, serial=0):
        self._serial = int(serial)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Union[Dater, int]) -> Dater:
        new_serial = self._serial.__add__(int(other))
        return type(self)(new_serial)

    __radd__ = __add__

    def __sub__(self, other: Union[Dater, int]) -> Union[Dater, int]:
        if isinstance(other, Dater):
            return self._serial - other._serial
        else:
            new_serial = self._serial.__sub__(int(other))
            return type(self)(new_serial)
    #)


class IntegerDater(Dater):
    #(
    _freq = Freq.INTEGER

    def __str__(self) -> str:
        serial = self._serial
        return self._freq.sdmx_format.format(serial=serial)
    #)


class DailyDater(Dater):
    #(
    _freq = Freq.DAILY

    @classmethod
    def serial_from_ymd(cls: type, year: int, month: int=1, day: int=1) -> int:
        return date(year, month, day).toordinal()

    @classmethod
    def construct_from_ymd(cls: type, *args):
        return cls(cls.serial_from_ymd(*args))

    @classmethod
    def ymd_from_serial(cls: type, serial: int) -> tuple(int, int, int):
        temp = date.fromordinal(serial)
        return temp.year, temp.month, temp.day

    @classmethod
    def yp_from_serial(cls: type, serial: int) -> tuple(int, int):
        boy_serial = date(date.fromordinal(serial).year, 1, 1)
        per = serial - boy_serial + 1
        year = date.fromordinal(serial).year
        return year, per

    @property
    def year_month_day(self) -> tuple(int, int, int):
        return self.ymd_from_serial(self._serial)

    @property
    def year_per(self) -> tuple(int, int):
        return self.yp_from_serial(self._serial)

    def __str__(self) -> str:
        year, month, day = self.year_month_day
        letter = self._freq.letter
        return self._freq.sdmx_format.format(year=year, month=month, day=day, letter=letter)
    #)


class RegularDater:
    #(
    @classmethod
    def serial_from_yp(cls: type, year: int, per: int) -> int:
        return int(year)*int(cls._freq.value) + int(per) - 1

    @classmethod
    def yp_from_serial(cls: type, serial: int) -> tuple(int, int):
        return serial//cls._freq.value, serial%cls._freq.value+1

    @classmethod
    def construct_from_yp(cls: type, year: int, per: int=1) -> type(cls):
        new_serial = cls.serial_from_yp(year, per)
        return cls(new_serial)

    @property
    def year_period(self) -> tuple(int, int):
        return self.yp_from_serial(self._serial)

    def __str__(self) -> str:
        year, per = self.year_period
        letter = self._freq.letter
        return self._freq.sdmx_format.format(year=year, per=per, letter=letter)
    #)


class YearlyDater(Dater, RegularDater):
    #(
    _freq: Freq = Freq.YEARLY
    #)


class HalfYearlyDater(Dater, RegularDater):
    #(
    _freq: Freq = Freq.HALF_YEARLY
    #)


class QuarterlyDater(Dater, RegularDater):
    #(
    _freq: Freq = Freq.QUARTERLY
    #)


class MonthlyDater(Dater, RegularDater):
    #(
    _freq: Freq = Freq.MONTHLY
    #)

yy = YearlyDater.construct_from_yp
hh = HalfYearlyDater.construct_from_yp
qq = QuarterlyDater.construct_from_yp
mm = MonthlyDater.construct_from_yp
ii = IntegerDater
dd = DailyDater.construct_from_ymd

