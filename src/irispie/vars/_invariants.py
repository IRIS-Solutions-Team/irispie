"""
"""


#[

from __future__ import annotations

import itertools as _it

from ..conveniences import descriptions as _descriptions
from .. import quantities as _quantities
from ..quantities import Quantity, QuantityKind
from ..dataslates import Dataslate
from ..fords.descriptors import SolutionVectors
from ..incidences.main import Token

from ._dimensions import Dimensions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable

#]


class Invariant(
    _descriptions.DescriptionMixin,
):
    r"""
    """
    #[

    __slots__ = (
        "quantities",
        "dimensions",
        "solution_vectors",
    )

    def __init__(
        self,
        endogenous_names: Iterable[str],
        exogenous_names: Iterable[str] | None = None,
        order: int = 1,
        intercept: bool = True,
    ) -> None:
        """
        """
        endogenous_names = tuple(endogenous_names)
        exogenous_names = tuple(exogenous_names) if exogenous_names else ()
        self.quantities = _create_quantities(endogenous_names, exogenous_names, )
        self.dimensions = Dimensions(
            num_endogenous=len(endogenous_names),
            order=order,
            has_intercept=intercept,
            num_exogenous=len(exogenous_names),
        )
        self._populate_solution_vectors()

    def _get_some_qids(self, kind: QuantityKind, ) -> tuple[int, ...]:
        r"""
        """
        return tuple(_quantities.generate_qids_by_kind(self.quantities, kind, ))

    def get_endogenous_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._get_some_qids(QuantityKind.TRANSITION_VARIABLE, )

    def get_residual_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._get_some_qids(QuantityKind.UNANTICIPATED_SHOCK, )

    def get_exogenous_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._get_some_qids(QuantityKind.EXOGENOUS_VARIABLE, )

    def _populate_solution_vectors(self, ) -> None:
        r"""
        """
        order = self.dimensions.order
        num_lagged_endogenous = self.dimensions.num_lagged_endogenous
        endogenous_qids = self.get_endogenous_qids()
        residual_qids = self.get_residual_qids()
        measurement_tokens = tuple(
            Token(qid=qid, shift=0, )
            for qid in endogenous_qids
        )
        transition_tokens = tuple(
            Token(qid=qid, shift=shift, )
            for shift, qid in _it.product(range(order), endogenous_qids, )
        )
        residual_tokens = tuple(
            Token(qid=qid, shift=0, )
            for qid in residual_qids
        )
        self.solution_vectors = SolutionVectors(
            transition_variables=transition_tokens,
            unanticipated_shocks=residual_tokens,
            anticipated_shocks=(),
            measurement_variables=measurement_tokens,
            measurement_shocks=(),
            true_initials=(True, ) * num_lagged_endogenous,
        )

    #]


def _create_quantities(
    endogenous_names: Iterable[str],
    exogenous_names: Iterable[str] | None = None,
) -> tuple[Quantity, ...]:
    """
    """
    quantities = []
    def append_quantities(names, kind, ):
        for n in names:
            quantities.append(Quantity(id=len(quantities), human=n, kind=kind, ))
    #
    endogenous_names = tuple(endogenous_names)
    exogenous_names = tuple(exogenous_names or ())
    residual_names = tuple(
        _residual_name_from_endogenous_name(name)
        for name in endogenous_names
    )
    #
    append_quantities(endogenous_names , QuantityKind.TRANSITION_VARIABLE, )
    append_quantities(exogenous_names, QuantityKind.EXOGENOUS_VARIABLE, )
    append_quantities(residual_names, QuantityKind.UNANTICIPATED_SHOCK, )
    #
    return tuple(quantities, )


_RESIDUAL_NAME_PREFIX = "res_"


def _residual_name_from_endogenous_name(n: str) -> str:
    r"""
    """
    return f"{_RESIDUAL_NAME_PREFIX}{n}"

