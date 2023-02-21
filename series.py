
#---{ External imports

from functools import partial

from IPython import embed
from numbers import Number

from collections.abc import (
    Iterable, Callable,
)

from typing import (
    Self, Protocol, runtime_checkable,
)

from copy import deepcopy 

from numpy import (
    ndarray as np_ndarray,
    full as np_full,
    nan as np_nan,
    vstack as np_vstack,
    hstack as np_hstack,
    copy as np_copy,
    ix_ as np_ix_,
    all as np_all,
    isnan as np_isnan,
    argmax as np_argmax,
    log as np_log,
    exp as np_exp,
    sqrt as np_sqrt,
    mean as np_mean,
    max as np_max,
)

#---}

#---{ Internal imports

from .dates import (
    Ranger, date_index, ResolvableProtocol,
)

#---}


def _str_row(date, data, date_str_format, numeric_format, nan_str: str):
    date_str = ("{:"+date_str_format+"}").format(date)
    value_format = "{:"+numeric_format+"}"
    data_str = "".join([value_format.format(v) for v in data])
    data_str = data_str.replace("nan", "{:>3}".format(nan_str))
    return date_str + data_str


def _get_date_positions(dates, base, num_periods):
    pos = list(date_index(dates, base))
    min_pos = min(pos)
    max_pos = max(pos)
    add_before = max(-min_pos, 0)
    add_after = max(max_pos - num_periods, 0)
    pos_adjusted = [p + add_before for p in pos]
    return pos_adjusted, add_before, add_after


def _trim_decorate(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self._trim()
    return wrapper


class Series:
    """
    """
    numeric_format: str = "15g"
    date_str_format: str = ">12"
    nan_str: str = "·"


    def __init__(self, /, num_columns=1, data_type=float, ):
        self.data_type = data_type
        self._reset(num_columns=num_columns, data_type=data_type)
        self.comment = ""


    def _reset(self, num_columns=None, data_type=None):
        num_columns = num_columns if num_columns else self.num_columns
        data_type = data_type if data_type else self.data_type
        self.start_date = None
        self.data = np_full((0, num_columns), np_nan, dtype=data_type)
        return self


    def _create_rows_of_nans(self, num_rows=0):
        return np_full((num_rows, self.shape[1]), np_nan, dtype=self.data_type)


    @property
    def shape(self):
        return self.data.shape


    @property
    def num_columns(self):
        return self.data.shape[1]


    @property
    def range(self):
        return Ranger(self.start_date, self.end_date) if self.start_date else None


    @property
    def end_date(self):
        return self.start_date + self.data.shape[0] - 1 if self.start_date else None


    @classmethod
    def from_data(cls, dates, data):
        self = cls()
        self.set_data(dates, data)


    @_trim_decorate
    def set_data(self, dates, data, columns=None) -> Self:
        if dates is Ellipsis:
            dates = Ranger(None, None)

        if isinstance(dates, ResolvableProtocol) and dates.needs_resolve:
            dates = dates.resolve(self)

        if not self.start_date:
            self.start_date = next(iter(dates))
            self.data = self._create_rows_of_nans(1)

        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before

        if columns is None:
            columns = slice(None) #self._get_default_columns()

        if isinstance(columns, slice):
            columns = range(*columns.indices(self.num_columns))

        if isinstance(pos, Iterable) and isinstance(columns, Iterable):
            if isinstance(data, tuple):
                for c, d in zip(columns, data):
                    self.data[pos, c] = d
            else:
                self.data[np_ix_(pos, columns)] = data
        else:
            self.data[pos, columns] = data

        return self


    def _get_default_columns(self):
        return 0 if self.data.shape[1]==1 else slice(None)


    def get_data(self, dates, columns=slice(None)):
        if dates is Ellipsis:
            dates = slice(None)

        if isinstance(dates, ResolvableProtocol) and dates.needs_resolve:
            dates = dates.resolve(self)

        if not dates:
            return self._create_rows_of_nans(0)[:,columns]

        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after)
        if isinstance(pos, Iterable) and isinstance(columns, Iterable):
            return data[np_ix_(pos, columns)]
        else:
            return data[pos, columns]


    def hstack(self, *args):
        if not args:
            return deepcopy(self)
        encompassing_range = self._get_encompassing_range(*args)
        new_data = self.get_data(encompassing_range)
        add_data = (x.get_data(encompassing_range) for x in args)
        new_data = np_hstack((new_data, *add_data))
        new = Series(num_columns=new_data.shape[1]);
        new.set_data(encompassing_range, new_data)
        return new


    def _get_encompassing_range(*args) -> Ranger:
        start_dates = [x.start_date for x in args if x.start_date]
        end_dates = [x.end_date for x in args if x.end_date]
        start_date = min(start_dates) if start_dates else None
        end_date = max(end_dates) if end_dates else None
        return Ranger(start_date, end_date)


    def _trim(self):
        if self.data.size==0:
            return self._reset()
        num_leading = _get_num_leading_nan_rows(self.data)
        if num_leading == self.data.shape[0]:
            return self._reset()
        num_trailing = _get_num_leading_nan_rows(self.data[::-1])
        if not num_leading and not num_trailing:
            return self
        slice_from = num_leading if num_leading else None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to), ...]
        if slice_from:
            self.start_date += int(slice_from)
        return self


    def _create_expanded_data(self, add_before, add_after):
        data = np_copy(self.data)
        if add_before:
            data = np_vstack((self._create_rows_of_nans(add_before), data))
        if add_after:
            data = np_vstack((data, self._create_rows_of_nans(add_after)))
        return data


    @property
    def _header_str(self):
        shape = self.shape
        return f"Series {shape[0]}-by-{shape[1]} {self.start_date}:{self.end_date}"


    @property
    def _data_str(self):
        if self.data.size>0:
            return "\n".join(
                _str_row(*z, self.date_str_format, self.numeric_format, self.nan_str) 
                for z in zip(self.range, self.data)
            )
        else:
            return None


    def __str__(self):
        header_str = self._header_str
        data_str = self._data_str
        all_str = "\n" + self._header_str + "\n"
        if data_str:
            all_str += "\n" + self._data_str
        return all_str


    def __repr__(self):
        return self.__str__()


    def _check_data_shape(self, data):
        if data.shape[1] != self.shape[1]:
            raise Exception("Time series data being assigned must preserve the number of columns")


    def __neg__(self):
        """
        -self
        """
        self.data = -self.data
        return self


    def __add__(self, other) -> Self|np_ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__add__(arg))


    def __radd__(self, other):
        """
        other + self
        """
        return self._unop(lambda data: data.__radd__(other))


    def __mul__(self, other) -> Self|np_ndarray:
        """
        self + other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__mul__(arg))


    def __rmul__(self, other):
        """
        other + self
        """
        return self._unop(lambda data: data.__rmul__(other))


    def __sub__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__sub__(arg))


    def __rsub__(self, other):
        """
        other - self
        """
        return self._unop(lambda data: data.__rsub__(other))


    def __truediv__(self, other):
        """
        self - other
        """
        arg = other.data if isinstance(other, type(self)) else other
        return self._unop(lambda data: data.__truediv__(arg))


    def __rtruediv__(self, other):
        """
        other - self
        """
        return self._unop(lambda data: data.__rtruediv__(other))



    def _unop(self, func: Callable, *args, **kwargs) -> Self|np_ndarray:
        new_data = func(self.data, *args, **kwargs)
        if new_data.shape == self.data.shape:
            new = deepcopy(self)
            new.data = new_data
            return new._trim()
        elif kwargs.get("axis")==1 and new_data.shape==(self.data.shape[0],):
            new = deepcopy(self)
            new.data = new_data.reshape(self.data.shape[0], 1)
            return new._trim()
        elif kwargs.get("axis") is None and new_data.shape==():
            return new_data
        elif kwargs.get("axis")==0 and new_data.shape==(self.data.shape[1],):
            return new_data
        else:
            raise Exception("Unary operation on a time series resulted in data with an unexpected shape")


    def _binop(self, other: Self|Number, func):
        if not isinstance(other, type(self)):
            unop_func = lambda data: func(data, other)
            return _unop(self, unop_func)


    @property
    def copy(self) -> Self:
        return deepcopy(self)



def _get_num_leading_nan_rows(data):
    try:
        num = next(x[0] for x in enumerate(data) if not np_all(np_isnan(x[1])))
    except StopIteration:
        num = data.shape[0]
    return num


def _unop(x: Series, func: Callable, *args, **kwargs) -> object:
    if isinstance(x, Series):
        return x._unop(func, *args, **kwargs)
    else:
        return func(x, *args, **kwargs)


log = partial(_unop, func=np_log)
exp = partial(_unop, func=np_exp)
sqrt = partial(_unop, func=np_sqrt)
mean = partial(_unop, func=np_mean)
max_ = partial(_unop, func=np_max)


def hstack(first, *args) -> Series:
    return first.hstack(*args)


