"""
"""


#[

from __future__ import annotations

from .. import dates as _dates
from ..dates import Period
from ..fords.descriptors import SolutionVectors
from ..simultaneous.main import Simultaneous
from .. import quantities as _quantities
from ..quantities import QuantityKind, Quantity
from ..incidences.main import Token

#]


class Invariant:
    """
    """
    #[

    __slots__ = (
        "num_periods",
        "quantities",
        "max_lag",
        "max_lead",
        "source_solution_vectors",
        "stacked_solution_vectors",
        "index_xi",
        "index_u",
        "index_v",
        "index_y",
        "index_w",
    )

    def __init__(self, /, **kwargs) -> None:
        """
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot, None, ), )

    @classmethod
    def from_simultaneous(
        klass,
        model: Simultaneous,
        num_periods: int,
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
        self.num_periods = num_periods
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
        #
        return self

    def _populate_indexes(self, /, ) -> None:
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

    def get_base_periods(self, start, / ) -> tuple[Period]:
        end = start + (self.num_periods - 1)
        return _dates.periods_from_until(start, end, )

    #]


def _create_index(source_vector, stacked_vector, /, ) -> list[int]:
    return [ source_vector.index(tok) for tok in stacked_vector ]


