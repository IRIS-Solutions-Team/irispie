"""
"""

#[
from __future__ import annotations

from .. import dates as _dates
from . import main as _series
#]


__all__ = (
    "shift",
    "diff", "difflog", "pct", "roc",
    "cum_diff", "cum_difflog", "cum_pct", "cum_roc",
)


def shift(x, by=-1, ) -> _series.Series:
    """
    """
    #[
    new = x.copy()
    new._shift(by)
    return new
    #]


def _negative_by_decorator(func):
    def wrapper(x, *args, **kwargs, ):
        if len(args) > 0:
            by = args[0]
            if not isinstance(by, str) and (int(by) != by or by >= 0):
                raise Exception("Time shift must be a negative integer")
        return func(x, *args, **kwargs, )
    return wrapper


@_negative_by_decorator
def diff(x, by=-1, /, ) -> _series.Series:
    return x - shift(x, by)


@_negative_by_decorator
def cum_diff(dx, by=-1, /, initial=0, range=_dates.Ranger(), ) -> _series.Series:
    return _cumulate(dx, by, "diff", initial, range, )


@_negative_by_decorator
def difflog(x, by=-1, /, ) -> _series.Series:
    return log(x) - log(shift(x, by))


@_negative_by_decorator
def cum_difflog(dx, by=-1, /, initial=1, range=_dates.Ranger(), ) -> _series.Series:
    return _cumulate(dx, by, "difflog", initial, range, )


@_negative_by_decorator
def pct(x, by=-1, /, ) -> _series.Series:
    return 100*(x/shift(x, by) - 1)


@_negative_by_decorator
def cum_pct(dx, by=-1, /, initial=1, range=_dates.Ranger(), ) -> _series.Series:
    return _cumulate(dx, by, "pct", initial, range, )


@_negative_by_decorator
def roc(x, by=-1, /, ) -> _series.Series:
    return x/shift(x, by)


@_negative_by_decorator
def cum_roc(roc, by=-1, /, initial=1, range=_dates.Ranger(), ) -> _series.Series | None:
    return _cumulate(roc, by, "roc", initial, range, )


_CUMULATIVE_FUNCTIONS = {
    "diff": {
        "forward": lambda x_past, change_curr: x_past + change_curr,
        "backward": lambda x_future, change_future: x_future - change_future,
    },
    "difflog": {
        "forward": lambda x_past, change_curr: x_past * exp(change_curr),
        "backward": lambda x_future, change_future: x_future / exp(change_future),
    },
    "pct": {
        "forward": lambda x_past, change_curr: x_past * (1 + change_curr/100),
        "backward": lambda x_future, change_future: x_future / (1 + change_future/100),
    },
    "roc": {
        "forward": lambda x_past, change_curr: x_past * change_curr,
        "backward": lambda x_future, change_future: x_future / change_future,
    },
}


def _cumulate(dx, by, func, initial, range, /, ) -> _series.Series:
    """
    """
    #[
    direction = range.direction
    cum_func = _CUMULATIVE_FUNCTIONS[func][direction]
    new = _series.Series(num_columns=dx.num_columns, data_type=dx.data_type, )
    match direction:
        case "forward":
            _cumulate_forward(new, dx, by, cum_func, initial, range, )
        case "backward":
            _cumulate_backward(new, dx, by, cum_func, initial, range, )
    return new.trim()
    #]


def _cumulate_forward(new, dx, by, cum_func, initial, range, /, ) -> None:
    """
    """
    #[
    range = range.resolve(dx)
    shifted_range = tuple(t.shift(by) for t in range)
    initial_range = _dates.Ranger(min(shifted_range), range.end_date, )
    new.set_data(initial_range, initial)
    for t, sh in zip(range, shifted_range):
        new_data = cum_func(new.get_data(sh, ), dx.get_data(t, ), )
        new.set_data(t, new_data)
    #]


def _cumulate_backward(new, dx, by, cum_func, initial, shifted_backward_range, /, ) -> None:
    """
    """
    #[
    dx_range_shifted = _dates.Ranger(dx.start_date, dx.end_date, -1, )
    dx_range_shifted.shift(by)
    shifted_backward_range = shifted_backward_range.resolve(dx_range_shifted)
    backward_range = shifted_backward_range.copy()
    backward_range.shift(-by)
    initial_range = _dates.Ranger(min(shifted_backward_range), backward_range.start_date)
    new.set_data(initial_range, initial)
    for t, sh in zip(backward_range, shifted_backward_range):
        new_data = cum_func(new.get_data(t, ), dx.get_data(t, ), )
        new.set_data(sh, new_data)
    #]


