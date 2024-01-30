"""
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Any, )

from .. import equations as _equations
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
        "all_names",
        "parameter_names",
        "_context",
        "_description",
    )

    def __init__(self, /, ) -> None:
        """
        """
        self.explanatories = ()
        self.all_names = ()
        self.parameter_names = ()
        self._context = {}
        self._description = ""

    @classmethod
    def from_equations(
        klass,
        equations: Iterable[_equations.Equation],
        quantities: Iterable[_quantities.Quantity] | None,
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
        self.collect_all_names()
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
    def lhs_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance.
        """
        return tuple( x.lhs_name for x in self.explanatories )

    @property
    def residual_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance
        """
        return tuple(
            x.residual_name
            for x in self.explanatories
            if x.residual_name is not None
        )

    def collect_all_names(
        self,
        /,
    ) -> None:
        """
        """
        all_names = set()
        for x in self.explanatories:
            all_names.update(x.all_names)
        rhs_only_names = tuple(all_names.difference(self.lhs_names + self.residual_names, ))
        self.all_names = self.lhs_names + rhs_only_names + self.residual_names

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
        order = Iterable[int],
        /,
    ) -> None:
        self.explanatories = [
            self.explanatories[i]
            for i in order
        ]
        self.collect_all_names()
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

