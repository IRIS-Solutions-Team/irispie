"""
Printers for iterative solvers
"""


#[

from __future__ import annotations

import os as _os
import numpy as _np
import scipy as _sp
import functools as _ft

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from numbers import Number
    from collections.abc import Iterable

#]


_NONE_SYMBOL = "•"
_CONTINUATION_SYMBOL = "⋯"
_LEN_CONTINUATION_SYMBOL = len(_CONTINUATION_SYMBOL)
_DEFAULT_PRINT_EVERY = 5
_DEFAULT_NORM_ORDER = 2
_ROUND_FOR_PRINT = 99

class EmptyIterPrinter:
    """
    Iterations printer for steady equator
    """
    #[

    _COLUMN_CLASSES = ()

    _HEADER_DIVIDER_CHAR = "-"
    _COLUMN_DIVIDER = " "*3

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        """
        # self._populate_equations(equations, )
        # self._populate_quantities(qids, qid_to_logly, qid_to_name, )
        self.reset()
        self.custom_header = kwargs.get("custom_header", None, )
        self._print_every = kwargs.get("print_every", _DEFAULT_PRINT_EVERY, )
        self._columns = tuple(i(**kwargs, ) for i in self._COLUMN_CLASSES)

    def reset(self, /, ) -> None:
        """
        """
        self._count = 0
        self.custom_header = None
        self._last_iter_string = None
        self.num_equations = None
        self.num_quantities = None

    def next(self, x, f, jacobian_calculated: bool, ) -> None:
        """
        Handle next function evaluation
        """
        if self._count == 0:
            self._first_iteration(x, f, jacobian_calculated, )
        for i in self._columns:
            i.next(self._count, x, f, jacobian_calculated, )
        self._last_iter_string = self._compose_iter_string()
        if self._count % self._print_every == 0:
            _print_to_width(self._last_iter_string)
        self._count += 1

    def _first_iteration(self, x, f, jacobian_calculated, ) -> None:
        self.num_equations = len(f)
        self.num_quantities = len(x)
        self.print_header()

    def print_header(self, /, ) -> None:
        """
        Print header for fuction evaluations
        """
        custom_header = str(self.custom_header or "")
        dimension_header = f"[Dimension {self.num_equations}×{self.num_quantities}]"
        header = self._COLUMN_DIVIDER.join(i.get_header_string() for i in self._columns)
        len_header = len(header)
        upper_line = self._HEADER_DIVIDER_CHAR + f"{custom_header}{dimension_header}" + self._HEADER_DIVIDER_CHAR * len_header
        upper_line = upper_line[:len_header]
        self._divider_line = self._HEADER_DIVIDER_CHAR * len_header
        _print_to_width("", upper_line, header, self._divider_line, )

    def conclude(self, /, ) -> None:
        """
        """
        self.print_footer()

    def print_footer(self, /, ) -> None:
        """
        Print footer for iterations
        """
        _print_to_width(self._divider_line, self._last_iter_string, self._divider_line, )

    def _compose_iter_string(self, /, ) -> str:
        return self._COLUMN_DIVIDER.join(i.get_iter_string() for i in self._columns)

    #]


class CounterColumn:
    #[

    _HEADER_SYMBOL = "ƒ-count"
    _WIDTH = 8

    def __init__(self, **kwargs, ) -> None:
        self._count = None

    def reset(self, /, ) -> None:
        self._count = None

    def next(self, count: int, x, f, jacobian_calculated: bool, /, ) -> None:
        self._count = count

    def get_header_string(self, /, ) -> str:
        return f"{self._HEADER_SYMBOL:>8}"

    def get_iter_string(self, /, ) -> str:
        return f"{self._count:{self._WIDTH}g}"

    #]


class FuncNormColumn:
    #[

    _HEADER_SYMBOL = "‖ƒ‖"
    _WIDTH_DECIMALS = 5
    _WIDTH_EXPONENT = 4
    _WIDTH = 2 + _WIDTH_DECIMALS + _WIDTH_EXPONENT

    def __init__(self, **kwargs, ) -> None:
        self._norm = None
        self._eval_norm = _ft.partial(_sp.linalg.norm, ord=kwargs.get("norm_order", _DEFAULT_NORM_ORDER), )

    def reset(self, /, ) -> None:
        self._norm = None

    def next(self, count: int, x, f, jacobian_calculated: bool, /, ) -> None:
        self._norm = self._eval_norm(f)

    def get_header_string(self, /, ) -> str:
        return f"{self._HEADER_SYMBOL:>{self._WIDTH}}"

    def get_iter_string(self, /, ) -> str:
        value_to_print = round(self._norm, _ROUND_FOR_PRINT, )
        return f"{value_to_print:.{self._WIDTH_DECIMALS}e}"

    #]


class JacobRecalculatedColumn:
    #[

    _HEADER_SYMBOL = "∇ƒ"
    _STATUS_SYMBOL = {True: "√", False: "×"}
    _WIDTH = 2

    def __init__(self, **kwargs, ) -> None:
        self._status = None

    def reset(self, /, ) -> None:
        self._status = None

    def next(self, count: int, x, f, jacobian_calculated: bool, /, ) -> None:
        self._status = jacobian_calculated

    def get_header_string(self, /, ) -> str:
        return f"{self._HEADER_SYMBOL:>{self._WIDTH}}"

    def get_iter_string(self, /, ) -> str:
        return f"{self._STATUS_SYMBOL[self._status]:>{self._WIDTH}}"

    #]


class _WorstColumn:
    #[

    _HEADER_SYMBOL = ...
    _NAME_SYMBOL = ...
    _WIDTH_NAME = ...

    _JOIN_SYMBOL = " "
    _WIDTH_JOIN_SYMBOL = len(_JOIN_SYMBOL)
    _WIDTH_DECIMALS = 5
    _WIDTH_EXPONENT = 4
    _WIDTH_NUMERIC = 2 + _WIDTH_DECIMALS + _WIDTH_EXPONENT

    def __init__(self, **kwargs, ) -> None:
        self._populate_names(**kwargs, )
        self._worst_value = None
        self._worst_name = None
        self._prev = None

    @property
    def _WIDTH(self, /, ) -> int:
        return self._WIDTH_NUMERIC + self._WIDTH_JOIN_SYMBOL + self._WIDTH_NAME

    def _populate_names(self, **kwargs, ) -> None:
        ...

    def _get_name(self, index, ) -> str:
        return (
            self._names[index % len(self._names)] if self._names
            else f"{self._NAME_SYMBOL}[{index}]"
        )

    def reset(self, /, ) -> None:
        self._worst_values = None
        self._worst_name = None
        self._prev = None

    def next(self, count: int, x, f, jacobian_calculated: bool, ) -> None:
        ...

    def get_header_string(self, /, ) -> str:
        header_numeric = f"{self._HEADER_SYMBOL:>{self._WIDTH_NUMERIC}}"
        return f"{header_numeric:<{self._WIDTH}}"

    def get_iter_string(self, /, ) -> str:
        return (
            self._get_proper_iter_string()
            if self._worst_value is not None
            else self._get_none_iter_string()
        )

    def _get_proper_iter_string(self, /, ) -> str:
        value_to_print = round(self._worst_value, _ROUND_FOR_PRINT, )
        return (
            f"{value_to_print:.{self._WIDTH_DECIMALS}e}"
            f"{self._JOIN_SYMBOL}"
            f"{self._worst_name:>{self._WIDTH_NAME}}"
        )

    def _get_none_iter_string(self, /, ) -> str:
        return (
            f"{_NONE_SYMBOL:>{self._WIDTH_NUMERIC}}"
            f"{' ':>{self._WIDTH_JOIN_SYMBOL}}"
            f"{' ':>{self._WIDTH_NAME}}"
        )

    #]


class WorstDiffXColumn(_WorstColumn, ):
    #[

    _HEADER_SYMBOL = "max|∆x|"
    _WIDTH_NAME = 10

    def _populate_names(self, **kwargs, ) -> None:
        self._names = tuple(
            _clip_string_exactly(i, self._WIDTH_NAME, )
            for i in kwargs["quantity_strings"]
        ) if "quantity_strings" in kwargs else None

    def next(self, count: int, x, f, jacobian_calculated: bool, ) -> None:
        self._worst_value = None
        self._worst_name = None
        if self._prev is not None:
            diff = abs(x - self._prev)
            index = _np.argmax(diff)
            self._worst_value = diff[index]
            self._worst_name = self._get_name(index, )
        self._prev = x

    #]


class WorstFuncColumn(_WorstColumn, ):
    #[

    _HEADER_SYMBOL = "max|ƒ|"
    _WIDTH_NAME = 25

    def _populate_names(self, **kwargs, ) -> None:
        self._names = tuple(
            _clip_string_exactly(i, self._WIDTH_NAME, )
            for i in kwargs["equation_strings"]
        ) if "equation_strings" in kwargs else None

    def next(self, count: int, x, f, jacobian_calculated: bool, ) -> None:
        abs_f = _np.abs(f)
        index = _np.argmax(abs_f)
        self._worst_value = abs_f[index]
        self._worst_name = self._get_name(index, )

    #]


def _clip_string_exactly(
    full_string: str,
    max_length: int,
    /,
) -> str:
    """
    Clip string to exactly to max length
    """
    return (
        full_string[:max_length-_LEN_CONTINUATION_SYMBOL] + _CONTINUATION_SYMBOL
        if len(full_string) > max_length
        else f"{full_string:<{max_length}}"
    )


def _print_to_width(*args, ) -> None:
    """
    """
    try:
        width = _os.get_terminal_size().columns
    except:
        width = None
    for text in args:
        if width is not None and len(text) > width:
            text = text[:width-1] + "⋯"
        print(text)


class IterPrinter(EmptyIterPrinter, ):
    """
    """
    #[

    _COLUMN_CLASSES = (
        CounterColumn,
        FuncNormColumn,
        JacobRecalculatedColumn,
        WorstDiffXColumn,
        WorstFuncColumn,
    )

    #]

