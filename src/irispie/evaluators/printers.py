"""
"""


#[
from collections.abc import (Iterable, )
from numbers import (Number, )
import numpy as _np
import scipy as _sp

from .. import (equations as _eq, quantities as _qu, )
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
        equations: _eq.Equations,
        qids: Iterable[int],
        qid_to_logly: dict[int, bool],
        qid_to_name: dict[int, str],
        /,
        every: int = 1,
    ) -> None:
        """
        """
        self._populate_equations(equations)
        self._populate_quantities(qids, qid_to_logly, qid_to_name, )
        ...
        self._iter_count = 0
        self._curr_f = None
        self._curr_x = None
        self._prev_x = None
        self._prev_f = None
        self._every = every

    def next(self, x, f, j_done, /, ) -> None:
        """
        Handle next iteration
        """
        self._curr_f = f.flatten()
        self._curr_x = x.flatten()
        f_norm = _sp.linalg.norm(self._curr_f, 2)
        if self._iter_count == 0:
            dimension = (self._curr_f.size, self._curr_x.size, )
            self.print_header(dimension, )
        self._last_iter_string = self.get_iter_string(f_norm, j_done, *self.find_worst_diff_x(), *self.find_worst_equation(), )
        if self._iter_count % self._every == 0:
            print(self._last_iter_string)
        self._prev_x = self._curr_x
        self._prev_f = self._curr_f
        self._curr_f = None
        self._curr_x = None
        self._iter_count += 1

    def find_worst_equation(self, /, ) -> tuple[Number, str]:
        """
        Find the largest function residual and the corresponding equation
        """
        index = _np.argmax(_np.abs(self._curr_f))
        worst_f = _np.abs(self._curr_f[index])
        worst_equation = self._equations[index].human if self._equations is not None else ""
        worst_equation = _clip_string_exactly(worst_equation, self._MAX_LEN_EQUATION_STRING)
        return f"{worst_f:.5e}", worst_equation

    def find_worst_diff_x(self, /, ) -> tuple[str, str]:
        """
        Find the largest change in x and the corresponding quantity
        """
        if self._prev_x is None:
            return f"{self._NAN_STRING:>11}", f"{self._NAN_STRING:{self._MAX_LEN_NAME_STRING}}"
        diff_x = self._curr_x - self._prev_x if self._prev_x is not None else None
        index = _np.argmax(diff_x)
        worst_diff_x = diff_x[index]
        worst_qid = self._qids[index]
        maybelog_worst_name = self._qid_to_print[worst_qid]
        maybelog_worst_name = _clip_string_exactly(maybelog_worst_name, self._MAX_LEN_NAME_STRING)
        return f"{worst_diff_x:.5e}", maybelog_worst_name

    def print_header(self, dimension, /, ) -> None:
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

    def print_footer(self, /, ) -> None:
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
    ) -> None:
        """
        Print info on current iteration
        """
        j_done_string = "√" if j_done else "×"
        return f"{self._iter_count:5g}   {f_norm:.5e}   {j_done_string:>5}   {worst_diff_x}   {worst_diff_x_name}   {worst_f}   {worst_equation}"

    def reset(self, /, ) -> None:
        """
        Reset iterations printer
        """
        self._iter_count = 0
        self._prev_x = None
        self._prev_f = None
    #]


class FlatSteadyIterPrinter(IterPrinter, ):
    """
    """
    #[
    def _populate_equations(
        self,
        equations: _eq.Equations,
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
            i: _qu.print_name_maybe_log(qid_to_name[i], qid_to_logly[i], )
            for i in qids
        }
    #]


class NonflatSteadyIterPrinter(IterPrinter, ):
    """
    """
    #[
    def _populate_equations(
        self,
        equations: _eq.Equations,
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
            i: _qu.print_name_maybe_log(qid_to_name[i], qid_to_logly[i], )
            for i in qids
        } | {
            -i: "∆" + _qu.print_name_maybe_log(qid_to_name[i], qid_to_logly[i], )
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
    return (
        full_string[:max_length-1] + "…" 
        if len(full_string) > max_length else f"{full_string:{max_length}}"
    )


