"""
Iteration printers for dynamic period-by-period systems
"""


#[

from __future__ import annotations

import neqs as _nq

from .. import quantities as _quantities

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..equations import Equation

#]


_ITER_PRINTER_COLUMNS = (
    "counter",
    "func_norm",
    "step_length",
    "jacob_status",
    "worst_diff_x",
    "worst_func",
)


def create_iter_printer(
    equations: Iterable[Equation],
    qids: Iterable[str],
    qid_to_logly: dict[str, bool],
    qid_to_name: dict[str, str],
    **kwargs,
) -> None:
    """
    """
    equation_strings = _get_equation_strings(equations, )
    quantity_strings = _get_quantity_strings(qids, qid_to_logly, qid_to_name, )
    return _nq.IterPrinter(
        equation_strings=equation_strings,
        quantity_strings=quantity_strings,
        columns=_ITER_PRINTER_COLUMNS,
        **kwargs,
    )


def _get_equation_strings(equations: Iterable[Equation], /, ) -> tuple[str, ...]:
    return tuple(i.human for i in equations)


def _get_quantity_strings(
    qids: Iterable[str],
    qid_to_logly: dict[str, bool],
    qid_to_name: dict[str, str],
    /,
) -> tuple[str, ...]:
    return tuple(
        _quantities.wrap_logly(qid_to_name[i], qid_to_logly.get(i, False), )
        for i in qids
    )

