"""
"""


#[
from __future__ import annotations

from typing import (Callable, )
from numbers import (Real, )
import numpy as _np

from .. import dates as _dates
from . import main as _series
from . import _functionalize
#]


__all__ = ()


class DiffMixin:
    """
    """
    #[

    def _shifted_op(
        self,
        shift: int | str,
        func: Callable,
        /,
    ) -> None:
        """
        """
        _catch_invalid_shift(shift, )
        other = self.copy()
        other.shift(shift, )
        self._binop(other, func, new=self, )

    def diff(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        """
        """
        self._shifted_op(shift, lambda x, y: x - y, )

    def diff_log(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        """
        """
        self._shifted_op(shift, lambda x, y: _np.log(x) - _np.log(y), )

    def roc(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        """
        """
        self._shifted_op(shift, lambda x, y: x/y, )

    def pct(
        self,
        shift: int | str = -1,
        /,
    ) -> None:
        """
        """
        self._shifted_op(shift, lambda x, y: 100*(x/y - 1), )

    #]


for n in ("diff", "diff_log", "pct", "roc", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


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


class CumMixin:
    """
    """
    #[

    def cum_diff(self, *args, **kwargs, ) -> None:
        """
        """
        self._cumulate("diff", *args, **kwargs, )

    def cum_diff_log(self, *args, **kwargs, ) -> None:
        """
        """
        self._cumulate("diff_log", *args, **kwargs, )

    def cum_pct(self, *args, **kwargs, ) -> None:
        """
        """
        self._cumulate("pct", *args, **kwargs, )

    def cum_roc(self, *args, **kwargs, ) -> None:
        """
        """
        self._cumulate("roc", *args, **kwargs, )

    def _cumulate(
        self,
        func_name: str,
        shift: int | str = -1,
        initial: Real | _series.Series | None = None,
        range: _dates.Ranger | None = None,
    ) -> None:
        """
        """
        _catch_invalid_shift(shift, )
        range = _dates.Ranger(None, None, ) if range is None else range
        range = range.resolve(self, )
        direction = range.direction
        factory = _CUMULATIVE_FACTORY[func_name]
        cum_func = factory[direction]
        initial = factory["initial"] if initial is None else initial
        if direction == "forward":
            self._cumulate_forward(shift, cum_func, initial, range, )
        elif direction == "backward":
            self._cumulate_backward(shift, cum_func, initial, range, )

    def _cumulate_forward(self, shift, cum_func, initial, range, /, ) -> None:
        """
        """
        shifted_range = tuple(t.shift(shift, ) for t in range)
        initial_range = _dates.Ranger(min(shifted_range), range.end_date, )
        orig = self.copy()
        self.empty()
        self.set_data(initial_range, initial)
        for t, sh in zip(range, shifted_range):
            new_data = cum_func(self.get_data(sh, ), orig.get_data(t, ), )
            self.set_data(t, new_data)

    def _cumulate_backward(self, shift, cum_func, initial, shifted_backward_range, /, ) -> None:
        """
        """
        orig_range_shifted = _dates.Ranger(self.start_date, self.end_date, -1, )
        orig_range_shifted.shift(shift, )
        shifted_backward_range = shifted_backward_range.resolve(orig_range_shifted, )
        backward_range = shifted_backward_range.copy()
        backward_range.shift(-shift, )
        initial_range = _dates.Ranger(min(shifted_backward_range), backward_range.start_date, )
        orig = self.copy()
        self.empty()
        self.set_data(initial_range, initial, )
        for t, sh in zip(backward_range, shifted_backward_range, ):
            new_data = cum_func(self.get_data(t, ), orig.get_data(t, ), )
            self.set_data(sh, new_data, )


for n in ("cum_diff", "cum_diff_log", "cum_pct", "cum_roc", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _catch_invalid_shift(shift: int | str, ):
    """
    """
    if not isinstance(shift, str) and (int(shift) != shift or shift >= 0):
        raise ValueError("Time shift must be a negative integer or a string")

