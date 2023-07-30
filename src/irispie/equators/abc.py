"""
Mixins and protocols for equator classes
"""


#[
import numpy as _np
from typing import (Protocol, Callable, )
from collections.abc import (Iterable, )

from .. import (equations as _eq, quantities as _qu, )
from ..aldi import (adaptations as _aa, )
#]


class EquatorProtocol(Protocol):
    min_shift: int
    max_shift: int
    _equations: _eq.Equations
    _xtrings: Iterable[str]
    _func: Callable


class EquatorMixin:
    """
    """
    #[
    @property
    def equations_human(self: EquatorProtocol, /, ) -> tuple[str]:
        return tuple(eqn.human for eqn in self._equations)

    @property
    def num_equations(self: EquatorProtocol, /, ) -> int:
        """
        """
        return len(self._equations)

    def _create_equator_function(
        self: EquatorProtocol,
        /,
        function_context: dict | None = None,
    ) -> None:
        """
        """
        function_context = _aa.add_function_adaptations_to_custom_functions(function_context)
        function_context["_array"] = _np.array
        self._xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = " , ".join(self._xtrings)
        self._func = eval(_eq.EVALUATOR_PREAMBLE + f"_array([{func_string}], dtype=float)", function_context)

    def _populate_min_max_shifts(self: EquatorProtocol, /, ) -> None:
        """
        """
        self.min_shift = _eq.get_min_shift_from_equations(self._equations)
        self.max_shift = _eq.get_max_shift_from_equations(self._equations)
    #]


