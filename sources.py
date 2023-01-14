"""
Model source
"""

#[
from __future__ import annotations

import collections

from typing import NoReturn
from collections.abc import Iterable

from .equations import (
    EquationKind, Equation,
)
from .quantities import (
    QuantityKind, Quantity,
)
#]


class Source:
    """
    """
    #[
    def __init__(self, /) -> NoReturn:
        self.quantities = []
        self.equations = []
        self.log_variables = []
        self.all_but = False
        self.sealed = False
        self.qid_to_logly = {}


    def seal(self, /) -> NoReturn:
        _check_unique_names(qty.human for qty in self.quantities)
        self._populate_logly()
        self.sealed = True


    @property
    def num_quantities(self, /) -> int:
        return len(self.quantities)


    @property
    def all_names(self, /) -> Iterable[str]:
        return [ qty.human for qty in self.quantities ]


    def add_parameters(self, names: Iterable[str] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.PARAMETER)


    def add_transition_variables(self, names: Iterable[str] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.TRANSITION_VARIABLE)


    def add_transition_shocks(self, names: Iterable[str] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.TRANSITION_SHOCK)


    def add_transition_equations(self, humans: Iterable[str] | None) -> NoReturn:
        self._add_equations(humans, EquationKind.TRANSITION_EQUATION)


    def add_measurement_variables(self, names: Iterable[str] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.MEASUREMENT_VARIABLE)


    def add_measurement_shocks(self, names: Iterable[str] | None) -> NoReturn:
        self._add_quantities(names, QuantityKind.MEASUREMENT_SHOCK)


    def add_measurement_equations(self, humans: Iterable[str] | None) -> NoReturn:
        self._add_equations(humans, EquationKind.MEASUREMENT_EQUATION)


    def add_log_variables(self, log_variables: Iterable[str] | None) -> NoReturn:
        self.log_variables += log_variables if log_variables else []


    def _add_quantities(self, names: Iterable[str] | None, kind: QuantityKind) -> NoReturn:
        if not names:
            return
        offset = len(self.quantities)
        self.quantities = self.quantities + [
            Quantity(id=id, human=name.strip(), kind=kind, logly=None) 
            for id, name in enumerate(names, start=offset)
        ]


    def _add_equations(self, humans: Iterable[str] | None, kind: EquationKind, /) -> NoReturn:
        if not humans:
            return
        offset = len(self.equations)
        self.equations = self.equations + [
            Equation(id=id, human=human.replace(" ", ""), kind=kind) 
            for id, human in enumerate(humans, start=offset) 
        ]


    def _populate_logly(self, /) -> NoReturn:
        default_logly = bool(self.all_but)
        qid_to_logly = { 
            qty.id: default_logly if qty.human not in self.log_variables else not default_logly
            for qty in self.quantities
            if qty.kind in QuantityKind.VARIABLE
        }
        quantities = self.quantities
        self.quantities = [
            Quantity(id=qty.id, human=qty.human, kind=qty.kind, logly=qid_to_logly.get(qty.id, None))
            for qty in quantities
        ]


    @classmethod
    def from_lists(
        cls,
        /,
        transition_variables: Iterable[str], 
        transition_equations: Iterable[str], 
        transition_shocks: Iterable[str] | None = None,
        measurement_variables: Iterable[str] | None = None,
        measurement_equations: Iterable[str] | None = None,
        measurement_shocks: Iterable[str] | None = None,
        parameters: Iterable[str] | None = None,
        log_variables: Iterable[str] | None = None,
        all_but: bool = False,
    ) -> Self:
        """
        """
        self = Source()
        self.add_transition_variables(transition_variables)
        self.add_transition_equations(transition_equations)
        self.add_transition_shocks(transition_shocks)
        self.add_measurement_variables(measurement_variables)
        self.add_measurement_equations(measurement_equations)
        self.add_measurement_shocks(measurement_shocks)
        self.add_parameters(parameters)
        self.all_but = all_but
        self.add_log_variables(log_variables)
        self.seal()
        return self
    #]



def _check_unique_names(names: Iterable[str]) -> NoReturn:
    """
    """
    #[
    name_counter = collections.Counter(names)
    if any(c>1 for c in name_counter.values()):
        duplicates = ( n for n, c in name_counter.items() if c>1 )
        raise Exception("Duplicate names " + ", ".join(duplicates))
    #]

