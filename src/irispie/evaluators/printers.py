"""
Iteration printers
"""


#[
from collections.abc import (Iterable, )
from numbers import (Number, )
import numpy as _np
import scipy as _sp
import os as _os

from .. import equations as _equations
from .. import quantities as _quantities
#]


class IterPrinter:
    """
    Iterations printer for steady equator
    """
    #[

    _NAN_STRING = "•"
    _HEADER_DIVIDER_CHAR = "-"
    _MAX_LEN_EQUATION_STRING = 20
    _MAX_LEN_NAME_STRING = 10

    def __init__(
        self, 
        equations: Iterable[_equations.Equation],
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
        qid_to_name: dict[int, str],
        /,
        every: int = 5,
        header_message: str | None = None,
    ) -> None:
        """
        """
        self._populate_equations(equations, )
        self._populate_quantities(qids, qid_to_logly, qid_to_name, )
        self.reset()
        self.header_message = header_message
        self._every = every

    def reset(self, /, ) -> None:
        """
        """
        self._func_count = 0
        self._curr_f = None
        self._curr_x = None
        self._prev_x = None
        self._prev_f = None
        self.header_message = None

    @property
    def num_equations(self, /, ) -> int:
        """
        """
        return len(self._equations)

    def next(self, x, f, /, jacobian_calculated, ) -> None:
        """
        Handle next function evaluation
        """
        self._curr_f = _np.array(tuple(f))
        self._curr_x = _np.array(tuple(x))
        f_norm = _sp.linalg.norm(self._curr_f, 2)
        if self._func_count == 0:
            dimension = (self._curr_f.size, self._curr_x.size, )
            self.print_header(dimension, )
        self._last_iter_string = self.get_iter_string(f_norm, jacobian_calculated, *self.find_worst_diff_x(), *self.find_worst_equation(), )
        if self._func_count % self._every == 0:
            _print_to_width(self._last_iter_string)
        self._prev_x = self._curr_x
        self._prev_f = self._curr_f
        self._curr_f = None
        self._curr_x = None
        self._func_count += 1

    def find_worst_equation(self, /, ) -> tuple[Number, str]:
        """
        Find the largest function residual and the corresponding equation
        """
        abs_curr_f = _np.abs(self._curr_f)
        index = _np.argmax(abs_curr_f)
        worst_f = abs_curr_f[index]
        worst_equation = self._equations[index].human if self._equations is not None else ""
        worst_equation = _clip_string_exactly(worst_equation, self._MAX_LEN_EQUATION_STRING)
        return f"{worst_f:.5e}", worst_equation

    def find_worst_diff_x(self, /, ) -> tuple[str, str]:
        """
        Find the largest change in x and the corresponding quantity
        """
        if self._prev_x is None:
            return f"{self._NAN_STRING:>11}", f"{self._NAN_STRING:{self._MAX_LEN_NAME_STRING}}"
        diff_x = abs(self._curr_x - self._prev_x) if self._prev_x is not None else None
        index = _np.argmax(diff_x)
        worst_diff_x = diff_x[index]
        worst_qid = self._qids[index]
        maybelog_worst_name = self._qid_to_print[worst_qid]
        maybelog_worst_name = _clip_string_exactly(maybelog_worst_name, self._MAX_LEN_NAME_STRING)
        return f"{worst_diff_x:.5e}", maybelog_worst_name

    def print_header(self, dimension, /, ) -> None:
        """
        Print header for fuction evaluations
        """
        top_line = f"{str(self.header_message or '')}[Dimension {dimension[0]}×{dimension[1]}]"
        header = f"{'ƒ-count':>8}   {'‖ƒ‖':>11}   {'∇ƒ':>5}   {'max|∆x|':>11}   {' ':10}   {'max|ƒ|':>11}   {''}"
        len_top_line = len(top_line)
        len_header = len(header) + self._MAX_LEN_EQUATION_STRING + 1
        self._divider_line = self._HEADER_DIVIDER_CHAR * max(len_header, len_top_line, )
        upper_divider = self._divider_line
        upper_divider = self._divider_line[0:1] + top_line + self._divider_line[1 + len(top_line):]
        lower_divider = self._divider_line
        _print_to_width("", upper_divider, header, lower_divider, )

    def print_footer(self, /, ) -> None:
        """
        Print footer for iterations
        """
        _print_to_width(self._divider_line, self._last_iter_string, self._divider_line, )

    def get_iter_string(
        self,
        f_norm: Number,
        jacobian_calculated: bool,
        worst_diff_x: str | None,
        worst_diff_x_name: str | None,
        worst_f: Number,
        worst_equation: str,
        /,
    ) -> None:
        """
        Print info on current iteration
        """
        j_done_string = "√" if jacobian_calculated else "×"
        return f"{self._func_count:8g}   {f_norm:.5e}   {j_done_string:>5}   {worst_diff_x}   {worst_diff_x_name}   {worst_f}   {worst_equation}"

    #]


class BasicIterPrinter(IterPrinter, ):
    """
    """
    #[
    def _populate_equations(
        self,
        equations: Iterable[_equations.Equation],
        /,
    ) -> None:
        """
        """
        self._equations = tuple(equations)

    def _populate_quantities(
        self,
        qids: Iterable[int],
        qid_to_logly: dict[str, bool],
        qid_to_name: dict[str, str],
        /,
    ) -> None:
        self._qids = tuple(qids)
        self._qid_to_print = {
            i: _quantities.wrap_logly(qid_to_name[i], qid_to_logly[i], )
            for i in qids
        }
    #]


FlatSteadyIterPrinter = BasicIterPrinter


class NonflatSteadyIterPrinter(IterPrinter, ):
    """
    """
    #[
    def _populate_equations(
        self,
        equations: Iterable[_equations.Equation],
        /,
    ) -> None:
        """
        """
        self._equations = tuple(equations) + tuple(equations)

    def _populate_quantities(
        self,
        qids: Iterable[str],
        qid_to_logly: dict[str, bool],
        qid_to_name: dict[str, str],
        /,
    ) -> None:
        self._qids = tuple(qids) + tuple(-i for i in qids)
        self._qid_to_print = {
            i: _quantities.wrap_logly(qid_to_name[i], qid_to_logly[i], )
            for i in qids
        } | {
            -i: "∆" + _quantities.wrap_logly(qid_to_name[i], qid_to_logly[i], )
            for i in qids
        }
    #]


class _FuncNormColumn:
    pass


def _clip_string_exactly(
    full_string: str,
    max_length: int,
    /,
) -> str:
    """
    Clip string to exactly to max length
    """
    return \
        full_string[:max_length-1] + "⋯" \
        if len(full_string) > max_length else f"{full_string:{max_length}}"


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

