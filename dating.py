
from __future__ import annotations
from datetime import date
from typing import Union, Optional
from enum import Enum
from collections.abc import Sequence, Iterable


class Freq(Enum):
    #(
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
    #)


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


class Date:
    #(
    freq = None

    def __init__(self, serial=0):
        self.serial = int(serial)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Union[int, Sequence[int]]) -> Union[Date, Sequence]:
        if isinstance(other, Sequence):
            return [ self.__add__(i) for i in other ]
        else:
            return type(self)(self.serial + int(other))

    __radd__ = __add__

    def __sub__(self, other: Union[Date, int, Sequence]) -> Union[Date, int, Sequence]:
        if isinstance(other, Range):
            return other.__rsub__(self)
        elif isinstance(other, Sequence):
            return [ self.__sub__(i) for i in other ]
        elif isinstance(other, Date):
            return self.serial - other.serial
        else:
            return type(self)(self.serial - int(other))

    def __rsub__(self, other: Sequence) -> Union[Date, int, Sequence]:
        if isinstance(other, Sequence):
            return [ i-self for i in other ]

    def __rshift__(start_date: Date, end_date: Date) -> Range:
        return Range(start_date, end_date, 1) 

    def __lshift__(start_date: Date, end_date: Date) -> Range:
        return Range(start_date, end_date, -1) 

    def __index__(self):
        return self.serial
    #)


class IntegerDate(Date):
    #(
    freq = Freq.INTEGER

    def __str__(self) -> str:
        serial = self.serial
        return self.freq.sdmx_format.format(serial=serial)
    #)


class DailyDate(Date):
    #(
    freq = Freq.DAILY

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

    def yp_from_serial(self) -> tuple(int, int):
        boy_serial = date(date.fromordinal(self.serial).year, 1, 1)
        per = self.serial - boy_serial + 1
        year = date.fromordinal(self.serial).year
        return year, per

    @property
    def year_month_day(self) -> tuple(int, int, int):
        return self.ymd_from_serial(self.serial)

    @property
    def year_per(self) -> tuple(int, int):
        return self.yp_from_serial()

    def __str__(self) -> str:
        year, month, day = self.year_month_day
        letter = self.freq.letter
        return self.freq.sdmx_format.format(year=year, month=month, day=day, letter=letter)
    #)


class RegularDate:
    #(
    @classmethod
    def serial_from_yp(cls: type, year: int, per: int) -> int:
        return int(year)*int(cls.freq.value) + int(per) - 1

    def yp_from_serial(self) -> tuple(int, int):
        return self.serial//self.freq.value, self.serial%self.freq.value+1

    @classmethod
    def construct_from_yp(cls: type, year: int, per: int=1) -> type(cls):
        new_serial = cls.serial_from_yp(year, per)
        return cls(new_serial)

    @property
    def year_period(self) -> tuple(int, int):
        return self.yp_from_serial()

    def __str__(self) -> str:
        year, per = self.year_period
        letter = self.freq.letter
        return self.freq.sdmx_format.format(year=year, per=per, letter=letter)
    #)


class YearlyDate(Date, RegularDate): freq: Freq = Freq.YEARLY
class HalfYearlyDate(Date, RegularDate): freq: Freq = Freq.HALF_YEARLY
class QuarterlyDate(Date, RegularDate): freq: Freq = Freq.QUARTERLY
class MonthlyDate(Date, RegularDate): freq: Freq = Freq.MONTHLY


yy = YearlyDate.construct_from_yp
hh = HalfYearlyDate.construct_from_yp
qq = QuarterlyDate.construct_from_yp
mm = MonthlyDate.construct_from_yp

ii = IntegerDate
dd = DailyDate.construct_from_ymd


class Range(Sequence):
    #(
    def __init__(self, start_date: Date, end_date: Optional[Date]=None, step: int=1) -> None:
        self._class = type(start_date)
        start_serial = start_date.serial
        end_serial = end_date.serial if end_date is not None else self.start_serial
        end_serial += 1 if step>0 else -1
        self.serials = range(start_serial, end_serial, step)

    @property
    def freq(self) -> Freq:
        return self._class.freq

    @property
    def start_date(self) -> Date:
        return self._class(self.serials[0])

    @property
    def end_date(self) -> Date:
        return self._class(self.serials[-1] + (1 if self.serials.step>0 else -1))

    @property
    def step(self) -> int:
        return self.serials.step

    def __len__(self) -> int:
        return len(self.serials)

    def __getitem__(self, key: int) -> Union[Range, Date]:
        if isinstance(key, slice):
            self.serials = self.serials[key]
            return self
        else:
            return self._class(self.serials[key])

    def __str__(self) -> str:
        if len(self)==0:
            return "Empty-Range"
        else:
            step_string = f", {self.step}" if self.step!=1 else ""
            return f"Range({self.start_date}, {self.end_date}{step_string})"

    __repr__ = __str__

    def __add__(self, offset: int) -> range:
        return Range(self.start_date+offset, self.end_date+offset, self.step)

    __radd__ = __add__

    def __sub__(self, offset: Union[Date, int]) -> Union[range, Rage]:
        if isinstance(offset, Date):
            return range(self.start_date-offset, self.end_date-offset, self.step)
        else:
            return Range(self.start_date-offset, self.end_date-offset, self.step)

    def __rsub__(self, other: Date) -> range:
        if isinstance(other, Date):
            return range(other-self.start_date, other-self.end_date, -self.step)
    #)


