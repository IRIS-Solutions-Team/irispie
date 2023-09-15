"""
Sequential models
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Any, Callable, )
import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from ..conveniences import copies as _copies
from ..conveniences import descriptions as _descriptions
from ..incidences import main as _incidences
from ..incidences import blazer as _blazer
from ..explanatories import main as explanatories
from ..plans import main as _plans

from . import _simulate as _simulate
#]


__all__ = (
    "Sequential",
)


class Sequential(
    _simulate.SimulateMixin,
    _sources.SourceMixin,
    _copies.CopyMixin,
    _descriptions.DescriptionMixin,
):
    """
    """
    #[
    __slots__ = (
        "explanatories",
        "all_names",
        "_description",
    )

    def __init__(
        self,
        /,
    ) -> None:
        self.explanatories = ()
        self.all_names = ()
        self._description = ""

    @classmethod
    def from_equations(
        cls,
        equations: Iterable[_equations.Equation],
        /,
        custom_functions: dict[str, Callable] | None = None,
    ) -> Self:
        """
        """
        self = cls()
        self.explanatories = tuple(
            explanatories.Explanatory(e, custom_functions=custom_functions, )
            for e in equations
        )
        self._collect_all_names()
        self._finalize_explanatories()
        return self

    @classmethod
    def from_source(
        cls,
        source: _sources.AlgebraicSource,
        /,
        context: dict[str, Callable] | None = None,
    ) -> Self:
        """
        """
        self = cls.from_equations(source.dynamic_equations, custom_functions=context, )
        return self

    @property
    def lhs_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance.
        """
        return tuple(
            x.lhs_name
            for x in self.explanatories
        )

    @property
    def res_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance.
        """
        return tuple(
            x.res_name
            for x in self.explanatories
            if x.res_name is not None
        )

    @property
    def rhs_only_names(self, /, ) -> set[str]:
        """
        Set of names of RHS variables no appearing on the LHS.
        """
        lhs_res_names = self.lhs_names + self.res_names
        return tuple(
            n for n in self.all_names
            if n not in lhs_res_names
        )

    @property
    def identity_index(self, /, ) -> tuple[int]:
        """
        Tuple of indices of identities
        """
        return tuple(
            i for i, x in enumerate(self.explanatories)
            if x.is_identity
        )

    @property
    def nonidentity_index(self, /, ) -> tuple[int]:
        """
        Tuple of indices of nonidentities
        """
        return tuple(
            i for i, x in enumerate(self.explanatories)
            if not x.is_identity
        )

    @property
    def equations(self, /, ) -> tuple[_equations.Equation]:
        """
        Tuple of equations in order of appearance
        """
        return tuple(
            x.equation
            for x in self.explanatories
        )

    @property
    def lhs_quantities(self, /, ) -> tuple[_quantities.Quantity]:
        """
        Tuple of LHS quantities in order of appearance
        """
        lhs_names = self.lhs_names
        kind = _quantities.QuantityKind.LHS_VARIABLE
        logly = False
        return tuple(
            _quantities.Quantity(qid, name, kind, logly, desc, )
            for (qid, name), desc in zip(enumerate(self.all_names), self.descriptions)
            if name in lhs_names
        )

    @property
    def num_equations(self, /, ) -> int:
        """
        Number of equations
        """
        return len(self.equations)

    @property
    def descriptions(self, /, ) -> tuple[str]:
        """
        """
        return tuple(
            x.equation.description
            for x in self.explanatories
        )

    @property
    def incidence_matrix(self, /, ) -> _np.ndarray:
        def _shift_test(tok: _incidences.Token) -> bool:
            return tok.shift == 0
        return _equations.create_incidence_matrix(
            self.equations,
            self.lhs_quantities,
            shift_test=_shift_test,
        )

    @property
    def min_shift(self, /, ) -> int:
        """
        """
        return min(
            x.min_shift
            for x in self.explanatories
        )

    @property
    def max_shift(self, /, ) -> int:
        """
        """
        return max(
            x.max_shift
            for x in self.explanatories
        )

    def reorder_equations(
        self,
        order = Iterable[int],
    ) -> None:
        self.explanatories = [
            self.explanatories[i]
            for i in order
        ]
        self._collect_all_names()
        self._finalize_explanatories()

    def sequentialize(
        self,
        /,
    ) -> tuple[int, ...]:
        """
        Reorder the model equations so that they can be solved sequentially
        """
        if self.is_sequential:
            return tuple(range(self.num_equations))
        eids_reordered = _blazer.sequentialize_strictly(self.incidence_matrix, )
        self.reorder_equations(eids_reordered, )
        return tuple(eids_reordered)

    @property
    def is_sequential(
        self,
        /,
    ) -> bool:
        """
        """
        return _blazer.is_sequential(self.incidence_matrix, )

    def print_equations(
        self,
        /,
        indent: int = 4,
        descriptions: bool = True,
        separator: str = "\n\n",
    ) -> str:
        """
        """
        return separator.join(
            x.print_equation(indent=indent, description=descriptions, )
            for x in self.explanatories
        )

    def _collect_all_names(
        self,
        /,
    ) -> None:
        """
        """
        all_names = set()
        for x in self.explanatories:
            all_names.update(x.all_names)
        lhs_res_names = self.lhs_names + self.res_names
        rhs_only_names = tuple(all_names.difference(lhs_res_names, ))
        self.all_names = self.lhs_names + rhs_only_names + self.res_names

    def create_qid_to_name(self, ) -> dict[int, str]:
        """
        """
        return {
            i: name
            for i, name in enumerate(self.all_names)
        }

    def create_name_to_qid(self, ) -> dict[str, int]:
        """
        """
        return {
            name: i
            for i, name in enumerate(self.all_names)
        }

    def _finalize_explanatories(
        self,
        /,
    ) -> None:
        """
        """
        name_to_qid = self.create_name_to_qid()
        for x in self.explanatories:
            x.finalize(name_to_qid, )

    def __str__(self, /, ) -> str:
        """
        """
        indented = " " * 4
        return "\n".join((
            f"{indented}Sequential object",
            f"{indented}Description: \"{self.get_description()}\"",
            f"{indented}Number of equations: {self.num_equations}",
            f"{indented}Number of [nonidentities, identities]: [{len(self.nonidentity_index)}, {len(self.identity_index)}]",
            f"{indented}Number of RHS-only names (excluding residuals): {len(self.rhs_only_names)}",
            f"{indented}[Min, Max] time shift: [{self.min_shift:+g}, {self.max_shift:+g}]",
        ))

    def precopy(self, /, ) -> None:
        """
        """
        for x in self.explanatories:
            x.precopy()

    #
    # ===== Implement SimulatableProtocol =====
    # This protocal is used in Dataslabs to prepare data arrays
    #

    def get_min_max_shifts(
        self,
        /,
    ) -> tuple[int, int]:
        """
        """
        return self.min_shift, self.max_shift

    def get_databank_names(
        self,
        plan: _plans.Plan | None = None,
        /,
    ) -> tuple[str]:
        """
        """
        all_names = self.all_names
        if plan is not None:
            plan_names = set(plan.collect_databank_names())
            extra_plan_names = tuple(plan_names.difference(all_names))
            all_names = all_names + extra_plan_names
        return all_names

    #
    # ===== Implement PlannableProtocol =====
    #

    def can_be_exogenized(self, /, ) -> tuple[str]:
        return tuple(
            n for i, n in enumerate(self.lhs_names)
            if not self.explanatories[i].is_identity
        )

    def can_be_exogenized_when_data(self, /, ) -> tuple[str]:
        return self.can_be_endogenized()

    def can_be_endogenized(self, /, ) -> None:
        return None
    #]


