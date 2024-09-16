"""
Printers for iterative steady-state solvers
"""


#[

from __future__ import annotations

from .. import quantities as _quantities
from .. import iter_printers as _iter_printers

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..equations import Equation

#]


class _IterPrinter(_iter_printers.IterPrinter, ):
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
            **kwargs,
        )

    def _get_equation_strings(
        self,
        equations: Iterable[Equation],
        /,
    ) -> tuple[str, ...]:
        return tuple(i.human for i in equations)

    def _get_quantity_strings(*args, **kwargs, ) -> tuple[str, ...]:
        raise NotImplementedError

    #]


class FlatIterPrinter(_IterPrinter, ):
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


class NonflatIterPrinter(_IterPrinter, ):
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

