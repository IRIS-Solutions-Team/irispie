"""
"""

#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Any, )

from ..equations import (Equation, )
from ..quantities import (Quantity, )
from .. import quantities as _quantities
from ..explanatories import main as _explanatories
from ..conveniences import descriptions as _descriptions
#]


class Invariant(
    _descriptions.DescriptionMixin,
):
    """
    """
    #[

    __slots__ = (
        "explanatories",
        "lhs_names",
        "residual_names",
        "rhs_only_names",
        "parameter_names",
        "_context",
        "__description__",
    )

    def __init__(self, /, ) -> None:
        """
        """
        self.explanatories = ()
        self.lhs_names = ()
        self.residual_names = ()
        self.rhs_only_names = ()
        self.parameter_names = ()
        self._context = {}
        self.__description__ = ""

    @classmethod
    def from_equations(
        klass,
        equations: Iterable[Equation],
        quantities: Iterable[Quantity] | None,
        /,
        context: dict[str, Any] | None = None,
        description: str | None = None,
        **kwargs,
    ) -> None:
        """
        """
        self = klass()
        self._context = context or {}
        self.set_description(description, )
        self.explanatories = tuple(
            _explanatories.Explanatory(e, context=self._context, **kwargs, )
            for e in equations
        )
        self.collect_names()
        self.finalize_explanatories()
        quantity_names = _quantities.generate_all_quantity_names(quantities, )
        self.parameter_names = tuple(n for n in quantity_names if n in self.all_names)
        return self

    @property
    def num_equations(self, /, ) -> int:
        """
        Number of equations.
        """
        return len(self.explanatories)

    @property
    def num_lhs_names(self, /, ) -> int:
        """
        Number of unique LHS names
        """
        return len(self.lhs_names)

    @property
    def equations(self, /, ) -> tuple[Equation]:
        return tuple( x.equation for x in self.explanatories )

    def collect_names(self, /, ) -> None:
        """
        """
        self.collect_lhs_names()
        self.collect_residual_names()
        self.collect_rhs_only_names()

    def collect_lhs_names(self, /, ) -> None:
        """
        Tuple of names of LHS variables in order of their first appearance in the equations
        """
        self.lhs_names = []
        for x in self.explanatories:
            if x.lhs_name not in self.lhs_names:
                self.lhs_names.append(x.lhs_name)
        self.lhs_names = tuple(self.lhs_names)

    def collect_residual_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance
        """
        self.residual_names = []
        for x in self.explanatories:
            if x.residual_name is None:
                continue
            if x.residual_name in self.residual_names:
                continue
            self.residual_names.append(x.residual_name, )
        self.residual_names = tuple(self.residual_names, )

    def collect_rhs_only_names(
        self,
        /,
    ) -> None:
        """
        """
        all_names = set()
        for x in self.explanatories:
            all_names.update(x.all_names, )
        self.rhs_only_names = tuple(all_names.difference(self.lhs_names + self.residual_names, ))

    @property
    def all_names(self, /, ) -> tuple[str]:
        """
        """
        return self.lhs_names + self.rhs_only_names + self.residual_names

    def finalize_explanatories(
        self,
        /,
    ) -> None:
        """
        """
        name_to_qid = self.create_name_to_qid()
        for x in self.explanatories:
            x.finalize(name_to_qid, )

    def reorder_equations(
        self,
        new_order = Iterable[int],
        /,
    ) -> None:
        """
        """
        # Check if new_order is a valid permutation
        if sorted(new_order) != list(range(self.num_equations)):
            raise ValueError("New equation order must be a permutation of integers from 0 to num_equations-1")
        #
        self.explanatories = [
            self.explanatories[i]
            for i in new_order
        ]
        # We need to recollect the names to make sure the LHS names and the
        # residual names are in the right order consistent with the new order
        # of the equations
        self.collect_names()
        self.finalize_explanatories()

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        """
        """
        return { name: i for i, name in enumerate(self.all_names) }

    def create_name_to_qid(self, /, ) -> dict[str, int]:
        """
        """
        return { name: i for i, name in enumerate(self.all_names) }

    #]

