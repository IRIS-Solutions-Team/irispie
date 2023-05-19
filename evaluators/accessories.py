"""m = 
"""


#[
from __future__ import annotations

import numpy as np_
import scipy as sp_
from typing import (Self, NoReturn, Callable, )
from collections.abc import (Iterable, )

from .. import (equations as eq_, quantities as qu_, )
from ..aldi import (adaptations as aa_, )
#]


class EvaluatorMixin:
    """
    """
    #[
    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self, /, ) -> int:
        """
        """
        return len(self._equations)

    def _create_evaluator_function(
        self,
        /,
        function_context: dict | None = None,
    ) -> NoReturn:
        """
        """
        function_context = aa_.add_function_adaptations_to_custom_functions(function_context)
        function_context["_array"] = np_.array
        self._xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = " , ".join(self._xtrings)
        self._func = eval(eq_.EVALUATOR_PREAMBLE + f"_array([{func_string}], dtype=float)", function_context)

    def _populate_min_max_shifts(self) -> NoReturn:
        """
        """
        self.min_shift = eq_.get_min_shift_from_equations(self._equations)
        self.max_shift = eq_.get_max_shift_from_equations(self._equations)
    #]


class IterPrinter:
    """
    Iterations printer for steady evaluator
    """
    #[
    _NAN_STRING = "•"
    _HEADER_DIVIDER_CHAR = "-"
    _MAX_LEN_EQUATION_STRING = 20
    _MAX_LEN_NAME_STRING = 10
    #
    __slots__ = (
        "_equations", "_quantities", "_every", "_iter_count", "_prev_x", "_prev_f", "_last_iter_string", "_divider_line",
    )
    def __init__(
        self, 
        equations: eq_.Equations,
        quantities: qu_.Quantities,
        /,
        every: int = 1,
    ) -> NoReturn:
        self._equations = equations
        self._quantities = quantities
        self._iter_count = 0
        self._prev_x = None
        self._prev_f = None
        self._every = every

    def next(self, x, f, j_done, /, ) -> NoReturn:
        """
        Handle next iteration
        """
        f_norm = sp_.linalg.norm(f, 2)
        diff_x = x - self._prev_x if self._prev_x is not None else None
        if self._iter_count == 0:
            dimension = (f.size, x.size, )
            self.print_header(dimension, )
        self._last_iter_string = self.get_iter_string(f_norm, j_done, *self.find_worst_diff_x(diff_x, ), *self.find_worst_equation(f, ), )
        if self._iter_count % self._every == 0:
            print(self._last_iter_string)
        self._prev_x = np_.copy(x)
        self._prev_f = np_.copy(f)
        self._iter_count += 1

    def find_worst_equation(self, f, /, ) -> tuple[Number, str]:
        """
        Find the largest function residual and the corresponding equation
        """
        index = np_.argmax(np_.abs(f))
        worst_f = np_.abs(f[index])
        worst_equation = self._equations[index].human if self._equations is not None else ""
        worst_equation = _clip_string_exactly(worst_equation, self._MAX_LEN_EQUATION_STRING)
        return f"{worst_f:.5e}", worst_equation

    def find_worst_diff_x(self, diff_x, /, ) -> tuple[str, str]:
        """
        Find the largest change in x and the corresponding quantity
        """
        if self._prev_x is None:
            return f"{self._NAN_STRING:>11}", f"{self._NAN_STRING:{self._MAX_LEN_NAME_STRING}}"
        #
        index = np_.argmax(diff_x)
        worst_diff_x = diff_x[index]
        worst_name = self._quantities[index].print_name_maybe_log() if self._quantities is not None else ' '
        worst_name = _clip_string_exactly(worst_name, self._MAX_LEN_NAME_STRING)
        return f"{worst_diff_x:.5e}", worst_name

    def print_header(self, dimension, /, ) -> NoReturn:
        """
        Print header for iterations
        """
        dim_string = f"Dimension: {dimension[0]}×{dimension[1]}"
        header = f"{'iter':>5}   {'‖ƒ‖':>11}   {'∇ƒ':>5}   {'max|∆x|':>11}   {' ':10}   {'max|ƒ|':>11}   {''}"
        len_header = len(header) + self._MAX_LEN_EQUATION_STRING + 1
        self._divider_line = self._HEADER_DIVIDER_CHAR * len_header
        upper_divider = self._divider_line
        upper_divider = self._divider_line[0:2] + dim_string + self._divider_line[2 + len(dim_string):]
        lower_divider = self._divider_line
        print("", upper_divider, header, lower_divider, sep="\n")

    def print_footer(self, /, ) -> NoReturn:
        """
        Print footer for iterations
        """
        print(self._divider_line, self._last_iter_string, self._divider_line, sep="\n")

    def get_iter_string(
        self,
        f_norm: Number,
        j_done: bool,
        worst_diff_x: str | None,
        worst_diff_x_name: str | None,
        worst_f: Number,
        worst_equation: str,
        /,
    ) -> NoReturn:
        """
        Print info on current iteration
        """
        j_done_string = "√" if j_done else "×"
        return f"{self._iter_count:5g}   {f_norm:.5e}   {j_done_string:>5}   {worst_diff_x}   {worst_diff_x_name}   {worst_f}   {worst_equation}"

    def reset(self, /, ) -> NoReturn:
        """
        Reset iterations printer
        """
        self._iter_count = 0
        self._prev_x = None
        self._prev_f = None
    #]


class _FuncNormColumn:
    pass

def _clip_string_exactly(full_string: str, max_length: int, /, ) -> str:
    """
    Clip string to exactly to max length
    """
    return (
        full_string[:max_length-1] + "…" 
        if len(full_string) > max_length else f"{full_string:{max_length}}"
    )


