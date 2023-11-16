"""
Sequential models
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, Iterator, )
from typing import (Self, Any, )
import numpy as _np

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from ..conveniences import copies as _copies
from ..conveniences import files as _files
from ..incidences import main as _incidences
from ..incidences import blazer as _blazer
from ..explanatories import main as _explanatories
from .. import iter_variants as _iter_variants

from . import _invariants as _invariants
from . import _variants as _variants
from . import _simulate as _simulate
from . import _assigns as _assigns

#]


__all__ = (
    "Sequential",
)


class Sequential(
    _simulate.SimulateMixin,
    _assigns.AssignMixin,
    _iter_variants.IterVariantsMixin,
    _copies.CopyMixin,
    _sources.SourceMixin,
    _files.FromFileMixin,
):
    """
    """
    #[

    __slots__ = (
        "_variants",
    )

    def __init__(
        self,
        /,
    ) -> None:
        self._invariant = _invariants.Invariant()
        self._variants = []

    @classmethod
    def from_equations(
        klass,
        equations: Iterable[_equations.Equation],
        /,
        quantities: Iterable[_quantities.Quantity] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = _invariants.Invariant.from_equations(equations, quantities, **kwargs, )
        initial_variant = _variants.Variant(self._invariant.parameter_names, )
        self._variants = [ initial_variant ]
        return self

    @classmethod
    def from_source(
        klass,
        source: _sources.ModelSource,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass.from_equations(
            source.dynamic_equations,
            quantities=source.quantities,
            **kwargs,
        )
        return self

    @property
    def lhs_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance
        """
        return self._invariant.lhs_names

    @property
    def res_names(self, /, ) -> tuple[str]:
        """
        Tuple of names of LHS variables in order of appearance.
        """
        return self._invariant.res_names

    @property
    def rhs_only_names(self, /, ) -> set[str]:
        """
        Set of names of RHS variables no appearing on the LHS.
        """
        lhs_res_names = self._invariant.lhs_names + self._invariant.res_names
        return tuple(
            n for n in self._invariant.all_names
            if n not in lhs_res_names
        )

    @property
    def identity_index(self, /, ) -> tuple[int]:
        """
        Tuple of indices of identities
        """
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if x.is_identity
        )

    @property
    def nonidentity_index(self, /, ) -> tuple[int]:
        """
        Tuple of indices of nonidentities
        """
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if not x.is_identity
        )

    @property
    def equations(self, /, ) -> tuple[_equations.Equation]:
        """
        Tuple of equations in order of appearance
        """
        return tuple(
            x.equation
            for x in self._invariant.explanatories
        )

    @property
    def lhs_quantities(self, /, ) -> tuple[_quantities.Quantity]:
        """
        Tuple of LHS quantities in order of appearance
        """
        lhs_names = self._invariant.lhs_names
        kind = _quantities.QuantityKind.LHS_VARIABLE
        logly = False
        return tuple(
            _quantities.Quantity(qid, name, kind, logly, desc, )
            for (qid, name), desc in zip(enumerate(self._invariant.all_names), self.descriptions)
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
            for x in self._invariant.explanatories
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
            for x in self._invariant.explanatories
        )

    @property
    def max_shift(self, /, ) -> int:
        """
        """
        return max(
            x.max_shift
            for x in self._invariant.explanatories
        )

    def reorder_equations(self, *args, **kwargs, ) -> None:
        """
        """
        self._invariant.reorder_equations(*args, **kwargs, )

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

    def iter_explanatories(self, /, ) -> Iterator[_explanatories.Explanatory]:
        """
        """
        yield from self._invariant.explanatories

    @property
    def is_sequential(
        self,
        /,
    ) -> bool:
        """
        """
        return _blazer.is_sequential(self.incidence_matrix, )

    def get_description(self, /, ) -> str:
        """
        """
        return self._invariant.get_description()

    def set_description(self, *args, **kwargs, ) -> None:
        """
        """
        self._invariant.set_description(*args, **kwargs, )

    def set_extra_databox_names(self, *args, **kwargs, ) -> None:
        """
        """
        self._invariant.set_extra_databox_names(*args, **kwargs, )

    def get_human_equations(
        self,
        /,
        descriptions: bool = True,
        separator: str = "\n\n",
    ) -> tuple[str]:
        """
        """
        return tuple(
            x.equation.human
            for x in self._invariant.explanatories
        )

    def create_qid_to_name(self, *args, **kwargs, ) -> dict[int, str]:
        """
        """
        return self._invariant.create_qid_to_name(*args, **kwargs, )

    def create_name_to_qid(self, *args, **kwargs, ) -> dict[str, int]:
        """
        """
        return self._invariant.create_name_to_qid(*args, **kwargs, )

    def __repr__(self, /, ) -> str:
        """
        """
        indented = " " * 4
        return "\n".join((
            f"",
            f"{self.__class__.__name__} model",
            f"Description: \"{self.get_description()}\"",
            f"|",
            f"| Number of variants: {self.num_variants}",
            f"| Number of equations: {self.num_equations}",
            f"| Number of [nonidentities, identities]: [{len(self.nonidentity_index)}, {len(self.identity_index)}]",
            f"| Number of RHS-only names (excluding residuals): {len(self.rhs_only_names)}",
            f"| [Min, Max] time shift: [{self.min_shift:+g}, {self.max_shift:+g}]",
            f"|",
        ))

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __getitem__(
        self,
        request: int,
        /,
    ) -> Self:
        """
        """
        return self.get_variant(request, )

    #
    # ===== Implement SlatableProtocol =====
    #

    def get_min_max_shifts(self, /, ) -> tuple[int, int]:
        """
        """
        return self.min_shift, self.max_shift

    def get_databox_names(self, /, ) -> tuple[str]:
        """
        """
        extra_databox_names = self._invariant.extra_databox_names or ()
        return tuple(self._invariant.all_names) + tuple(extra_databox_names)

    def get_fallbacks(self, /, ) -> dict[str, Any]:
        """
        """
        fallbacks = { n: 0 for n in tuple(self._invariant.res_names) }
        fallbacks.update(self.get_parameters(), )
        return fallbacks

    def get_overwrites(self, /, ) -> dict[str, Any]:
        """
        """
        return {}

    def create_qid_to_logly(self, /, ) -> dict[str, bool]:
        """
        """
        return {}

    #
    # ===== Implement PlannableSimulateProtocol =====
    #

    @property
    def simulate_can_be_exogenized(self, /, ) -> tuple[str, ...]:
        return tuple(
            i.lhs_name for i in self._invariant.explanatories
            if not i.is_identity
        )

    simulate_can_be_when_data = simulate_can_be_exogenized
    simulate_can_be_endogenized = ()
    simulate_can_be_anticipated = ()

    #]


