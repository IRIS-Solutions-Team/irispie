"""
"""

#[
from __future__ import annotations

from typing import (Self, NoReturn, )

from .. import (equations as eq_, quantities as qu_, evaluators as ev_, )
from ..models import (facade as mf_, )
from ..fords import (descriptors as de_, )
#]

class Invariant:
    """
    Invariant part of a Model object
    """
    #[
    def __init__(self, model_source, **kwargs) -> Self:
        self._flags = mf_.ModelFlags.from_kwargs(**kwargs, )
        self._quantities = model_source.quantities[:]
        self._dynamic_equations = model_source.dynamic_equations[:]
        self._steady_equations = model_source.steady_equations[:]

        name_to_qid = qu_.create_name_to_qid(self._quantities)
        eq_.finalize_dynamic_equations(self._dynamic_equations, name_to_qid)
        eq_.finalize_steady_equations(self._steady_equations, name_to_qid)
        #
        self._dynamic_descriptor = de_.Descriptor(self._dynamic_equations, self._quantities)
        self._steady_descriptor = de_.Descriptor(self._steady_equations, self._quantities)
        #
        self._plain_evaluator_for_dynamic_equations = ev_.PlainEvaluator(
            eq_.generate_equations_of_kind(self._dynamic_equations, eq_.EquationKind.STEADY_EVALUATOR),
        )
        #
        self._plain_evaluator_for_steady_equations = ev_.PlainEvaluator(
            eq_.generate_equations_of_kind(self._steady_equations, eq_.EquationKind.STEADY_EVALUATOR),
        )
        #
        self._populate_min_max_shifts()

    def _populate_min_max_shifts(self) -> NoReturn:
        """
        """
        self._min_shift = eq_.get_min_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )
        self._max_shift = eq_.get_max_shift_from_equations(
            self._dynamic_equations + self._steady_equations
        )
    #]


