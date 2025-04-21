"""
"""


#[

from __future__ import annotations

from .. import dates as _dates
from ..dates import Period
from ..fords.descriptors import SolutionVectors, Squid
from ..simultaneous.main import Simultaneous
from .. import quantities as _quantities
from ..quantities import QuantityKind, Quantity
from ..incidences.main import Token

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Self, Any

#]


class Invariant:
    r"""
    """
    #[

    __slots__ = (
        "base_periods",
        "transition_period_indexes",
        "measurement_period_indexes",
        "quantities",
        "max_lag",
        "max_lead",
        "source_solution_vectors",
        "stacked_solution_vectors",
        "squid",
        "index_xi",
        "index_u",
        "index_v",
        "index_y",
        "index_w",
    )

    def __init__(self, **kwargs, ) -> None:
        """
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, None, ), )

    @classmethod
    def from_simultaneous(
        klass,
        model: Simultaneous,
        span: Iterable[Period],
        *,
        transition_variables: None | Iterable[str] = None,
        anticipated_shocks: None | Iterable[str] = None,
        unanticipated_shocks: None | Iterable[str] = None,
        measurement_variables: None | Iterable[str] = None,
        measurement_shocks: None | Iterable[str] = None,
    ) -> Self:
        """
        """
        self = klass()
        self._populate_periods(span, )
        self.quantities = model.quantities
        self.max_lag = model.max_lag
        self.max_lead = model.max_lead
        self.source_solution_vectors = model._invariant.dynamic_descriptor.solution_vectors
        #
        name_to_qid = model.create_name_to_qid()
        quantity_names = {
            "transition_variables": transition_variables,
            "anticipated_shocks": anticipated_shocks,
            "unanticipated_shocks": unanticipated_shocks,
            "measurement_variables": measurement_variables,
            "measurement_shocks": measurement_shocks,
        }
        for kind in quantity_names.keys():
            if quantity_names[kind] is None:
                quantity_names[kind] = model.get_names(kind=QuantityKind.from_keyword(kind, ), )
        #
        quantity_ids = { kind: None for kind in quantity_names.keys() }
        self.stacked_solution_vectors = SolutionVectors()
        for kind in quantity_names.keys():
            sorted_qids = sorted(set(tuple(name_to_qid[name] for name in quantity_names[kind])))
            vector = tuple( Token(qid, 0, ) for qid in sorted_qids )
            setattr(self.stacked_solution_vectors, kind, vector, )
        #
        self._populate_indexes()
        self._populate_squid(model, )
        #
        return self

    @property
    def num_periods(self, ) -> int:
        r"""
        """
        return len(self.base_periods)

    def _populate_periods(self, span: Iterable[Period], ) -> None:
        r"""
        """
        span = tuple(i for i in span)
        self.base_periods = _dates.periods_from_until(span[0], span[-1], )

    def _populate_indexes(self, ) -> None:
        self.index_xi = _create_index(
            self.source_solution_vectors.transition_variables,
            self.stacked_solution_vectors.transition_variables,
        )
        self.index_u = _create_index(
            self.source_solution_vectors.unanticipated_shocks,
            self.stacked_solution_vectors.unanticipated_shocks,
        )
        self.index_v = _create_index(
            self.source_solution_vectors.anticipated_shocks,
            self.stacked_solution_vectors.anticipated_shocks,
        )
        self.index_y = _create_index(
            self.source_solution_vectors.measurement_variables,
            self.stacked_solution_vectors.measurement_variables,
        )
        self.index_w = _create_index(
            self.source_solution_vectors.measurement_shocks,
            self.stacked_solution_vectors.measurement_shocks,
        )

    def _populate_squid(self, model, ) -> None:
        r"""
        """
        squid = Squid.from_squidable(model, )
        squid.y_qids = _extract_by_index(squid.y_qids, self.index_y, )
        squid.u_qids = _extract_by_index(squid.u_qids, self.index_u, )
        squid.v_qids = _extract_by_index(squid.v_qids, self.index_v, )
        squid.w_qids = _extract_by_index(squid.w_qids, self.index_w, )
        squid.std_u_qids = _extract_by_index(squid.std_u_qids, self.index_u, )
        squid.std_v_qids = _extract_by_index(squid.std_v_qids, self.index_v, )
        squid.std_w_qids = _extract_by_index(squid.std_w_qids, self.index_w, )
        self.squid = squid

    #]


def _create_index(source_vector, stacked_vector, ) -> list[int, ...]:
    r"""
    Return index of positions in source_vector for each token in stacked_vector.
    Needs to be a list, not a tuple, because it is used for indexing in numpy.
    """
    return [
        source_vector.index(tok)
        for tok in stacked_vector
    ]


def _extract_by_index(
    source: Sequence[Any],
    index: list[int],
) -> list[Any, ...]:
    r"""
    Return a list of elements from a source sequence at positions given by index
    """
    return [ source[i] for i in index ]

