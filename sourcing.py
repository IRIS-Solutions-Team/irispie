"""
Model source
"""


from typing import (
    Optional as tp_Optional, Iterable as tp_Iterable
)


from collections import (
    Counter as cl_Counter,
)


from .equations import (
    EquationKind, Equation,
)


from .quantities import (
    QuantityKind, Quantity,
)



class Source:
    """
    """
    def __init__(self):
        self.quantities: list[Quantity] = []
        self.equations: list[Equation] = []
        self.sealed = False

    def seal(self):
        _check_unique_names(qty.human for qty in self.quantities)
        self.sealed = True

    @property
    def id_to_log_flag(self) -> dict[int, bool]:
        return { q.id: q.log_flag for q in self.quantities }

    @property
    def num_quantities(self) -> int:
        return len(self.quantities)

    @property
    def all_names(self) -> list[str]:
        return [ qty.human for qty in self.quantities ]

    def add_parameters(self, names: tp_Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.PARAMETER)

    def add_transition_variables(self, names: tp_Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_VARIABLE)

    def add_transition_shocks(self, names: tp_Optional[list[str]]) -> None:
        self._add_quantities(names, QuantityKind.TRANSITION_SHOCK)

    def add_transition_equations(self, humans: tp_Optional[list[str]]) -> None:
        self._add_equations(humans, EquationKind.TRANSITION_EQUATION)


    def _add_quantities(self, names: tp_Optional[list[str]], kind: QuantityKind) -> None:
        if not names:
            return
        offset = len(self.quantities)
        self.quantities = self.quantities + [
            Quantity(id=id, human=name.replace(" ", ""), kind=kind) for id, name in enumerate(names, start=offset)
        ]


    def _add_equations(self, humans: tp_Optional[list[str]], kind: EquationKind) -> None:
        if not humans:
            return
        offset = len(self.equations)
        self.equations = self.equations + [ Equation(id=id, human=human.replace(" ", ""), kind=kind) for id, human in enumerate(humans, start=offset) ]
    #)



def _check_unique_names(names: tp_Iterable[str]) -> None:
    name_counter = cl_Counter(names)
    if any( c>1 for c in name_counter.values() ):
        duplicates = ( n for n, c in name_counter.items() if c>1 )
        raise Exception("Duplicate names " + ", ".join(duplicates))


