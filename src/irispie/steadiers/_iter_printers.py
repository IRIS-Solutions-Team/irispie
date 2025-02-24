"""
Printers for iterative steady-state solvers
"""


#[

from __future__ import annotations

from neqs import IterPrinter

from .. import quantities as _quantities

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..equations import Equation

#]


_COLUMNS = (
    "counter",
    "func_norm",
    "step_length",
    "jacob_status",
    "worst_diff_x",
    "worst_func",
)


class _IterPrinterWithStrings(IterPrinter, ):
    """
    """
    #[

    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        qids: Iterable[str],
        qid_to_logly: dict[str, bool],
        qid_to_name: dict[str, str],
        **kwargs,
    ) -> None:
        """
        """
        equation_strings = self._get_equation_strings(equations, )
        quantity_strings = self._get_quantity_strings(qids, qid_to_logly, qid_to_name, )
        super().__init__(
            equation_strings=equation_strings,
            quantity_strings=quantity_strings,
            columns=_COLUMNS,
            **kwargs,
        )

    def _get_equation_strings(
        self,
        equations: Iterable[Equation],
        /,
    ) -> tuple[str, ...]:
        return tuple(i.human for i in equations)

    def _get_quantity_strings(*args, **kwargs, ) -> tuple[str, ...]:
        ...

    #]


class FlatIterPrinter(_IterPrinterWithStrings, ):
    """
    """
    #[

    def _get_quantity_strings(
        self,
        qids: Iterable[str],
        qid_to_logly: dict[str, bool],
        qid_to_name: dict[str, str],
        /,
    ) -> tuple[str, ...]:
        return tuple(
            _quantities.wrap_logly(qid_to_name[i], qid_to_logly.get(i, False), )
            for i in qids
        )

    #]


class NonflatIterPrinter(_IterPrinterWithStrings, ):
    """
    """
    #[

    def _get_quantity_strings(
        self,
        qids: Iterable[str],
        qid_to_logly: dict[str, bool],
        qid_to_name: dict[str, str],
        /,
    ) -> tuple[str, ...]:
        return tuple(
            _quantities.wrap_logly(qid_to_name[i], qid_to_logly.get(i, False), )
            for i in qids
        ) + tuple(
            "∆" + _quantities.wrap_logly(qid_to_name[i], qid_to_logly.get(i, False), )
            for i in qids
        )

    #]

