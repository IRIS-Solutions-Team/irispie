
from functools import partial as ft_partial


from typing import (
    Iterable, Callable, Self,
    Protocol, runtime_checkable
)

from copy import deepcopy as cp_deepcopy

from numpy import (
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
)


from .dating import (
    Ranger as dt_Ranger,
    date_index as dt_date_index,
    ResolvableP as dt_ResolvableP,
)


def _str_row(date, data, date_str_format, numeric_format, nan_str: str):
    date_str = ("{:"+date_str_format+"}").format(date)
    value_format = "{:"+numeric_format+"}"
    data_str = "".join([value_format.format(v) for v in data])
    data_str = data_str.replace("nan", "{:>3}".format(nan_str))
    return date_str + data_str


def _get_date_positions(dates, base, num_periods):
    pos = list(dt_date_index(dates, base))
    min_pos = min(pos)
    max_pos = max(pos)
    add_before = max(-min_pos, 0)
    add_after = max(max_pos - num_periods, 0)
    pos_adjusted = [p + add_before for p in pos]
    return pos_adjusted, add_before, add_after


def _trim_decorate(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._trim()
    return wrapper


class Series:
    numeric_format = '15g'
    date_str_format = '>12'
    nan_str = "·"

    def __init__(self, num_columns=None, data_type=float):
        self._reset(num_columns, data_type)
        self.comment = ""


    def _reset(self, num_columns, data_type):
        num_columns = num_columns if num_columns else 1
        self.start_date = None
        self.data = np_full((0, num_columns), np_nan, dtype=data_type)


    def _create_nans(self, num_rows):
        return np_full((num_rows, self.shape[1]), np_nan, dtype=self.data_type)


    @property
    def shape(self):
        return self.data.shape


    @property
    def data_type(self):
        return self.data.dtype


    @property
    def range(self):
        return dt_Ranger(self.start_date, self.end_date) if self.start_date else None


    @property
    def end_date(self):
        return self.start_date + self.data.shape[0] - 1 if self.start_date else None


    @classmethod
    def from_data(cls, dates, data):
        self = cls()
        self.set_data(dates, data)


    @_trim_decorate
    def set_data(self, dates, data, columns=None) -> None:
        if dates is Ellipsis:
            dates = dt_Ranger(None, None)

        if isinstance(dates, dt_ResolvableP) and not dates.is_resolved:
            dates = dates.resolve(self)

        if not self.start_date:
            self.start_date = next(iter(dates))
            self.data = self._create_nans(1)

        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        self.data = self._create_expanded_data(add_before, add_after)
        if add_before:
            self.start_date -= add_before

        if columns is None:
            columns = self._get_default_columns()

        if isinstance(pos, Iterable) and isinstance(columns, Iterable):
            self.data[np_ix_(pos, columns)] = data
        else:
            self.data[pos, columns] = data

        return self


    def _get_default_columns(self):
        return 0 if self.data.shape[1]==1 else slice(None)


    def get_data(self, dates, columns=slice(None)):
        if dates is Ellipsis:
            dates = slice(None)

        if isinstance(dates, dt_ResolvableP) and not dates.is_resolved:
            dates = dates.resolve(self)

        if not dates:
            return self._create_nans(0)[:,columns]

        pos, add_before, add_after = _get_date_positions(dates, self.start_date, self.shape[0]-1)
        data = self._create_expanded_data(add_before, add_after)
        if isinstance(pos, Iterable) and isinstance(columns, Iterable):
            return data[np_ix_(pos, columns)]
        else:
            return data[pos, columns]


    def cat(self, *args):
        if not args:
            return cp_deepcopy(self)
        encompassing_range = self._get_encompassing_range(*args)
        new_data = self.get_data(encompassing_range)
        add_data = (x.get_data(encompassing_range) for x in args)
        new_data = np_hstack((new_data, *add_data))
        new = Series(num_columns=new_data.shape[1]);
        new.set_data(encompassing_range, new_data)
        return new


    def _get_encompassing_range(*args) -> dt_Ranger:
        start_dates = [x.start_date for x in args if x.start_date]
        end_dates = [x.end_date for x in args if x.end_date]
        start_date = min(start_dates) if start_dates else None
        end_date = max(end_dates) if end_dates else None
        return dt_Ranger(start_date, end_date)


    def _trim(self):
        if self.data.size==0:
            return
        num_leading = _get_num_leading_nan_rows(self.data)
        if num_leading == self.data.shape[0]:
            self.start_date = None
            return
        num_trailing = _get_num_leading_nan_rows(self.data[::-1])
        if not num_leading and not num_trailing:
            return
        slice_from = num_leading if num_leading else None
        slice_to = -num_trailing if num_trailing else None
        self.data = self.data[slice(slice_from, slice_to), ...]
        if slice_from:
            self.start_date += int(slice_from)


    def _create_expanded_data(self, add_before, add_after):
        data = np_copy(self.data)
        if add_before:
            data = np_vstack((self._create_nans(add_before), data))
        if add_after:
            data = np_vstack((data, self._create_nans(add_after)))
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


    def _unop(self, func: Callable, *args, **kwargs) -> Self:
        result = cp_deepcopy(self)
        result.data = func(result.data, *args, **kwargs)
        return result


    @property
    def copy(self) -> Self:
        return cp_deepcopy(self)



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


log = ft_partial(_unop, func=np_log)
exp = ft_partial(_unop, func=np_exp)
sqrt = ft_partial(_unop, func=np_sqrt)


def cat(first, *args) -> Series:
    return first.cat(*args)


