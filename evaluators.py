"""
"""

#[
from __future__ import annotations
from IPython import embed
import enum
import numpy
from numpy import (log, exp, )
from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )

from . import models
from . import quantities
from . import equations
from .functions import *
#]


class EvaluatorKind(enum.Flag, ):
    """
    """
    #[
    STEADY = enum.auto()
    STACKED = enum.auto()
    PERIOD = enum.auto()
    #]


class SteadyEvaluator:
    """
    """
    #[
    kind = EvaluatorKind.STEADY

    def __init__(
        self,
        model: models.core.Model,
        /,
        variant_id: int = 0
    ) -> NoReturn:
        self._equations = []
        self._quantities = []
        self._populate_equations(model)
        self._populate_quantities(model)
        self._qids = list(quantities.generate_all_qids(self._quantities))
        min_shift = equations.get_min_shift_from_equations(self._equations)
        max_shift = equations.get_max_shift_from_equations(self._equations)
        self._t_zero = -min_shift
        self._create_evaluator_function()
        self._create_incidence_matrix()
        num_columns = self._t_zero + 1 + max_shift

        variant = model._variants[variant_id]
        self._x: numpy.ndarray = model.create_steady_array(variant, num_columns=num_columns, shift_in_first_column=min_shift)
        self._z0: numpy.ndarray = self._z_from_x()


    @property
    def init(self, /) -> numpy.ndarray:
        return numpy.copy(self._z0)


    @property
    def quantities_human(self, /) -> Iterable[str]:
        return [ qty.human for qty in self._quantities ]


    @property
    def equations_human(self, /) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]


    @property
    def num_equations(self, /) -> int:
        return len(self._equations)


    @property
    def num_quantities(self, /) -> int:
        return len(self._quantities)


    def eval(self, current: numpy.ndarray | None = None, /):
        """
        """
        x = self._x_from_z(current)
        return self._func(x)


    def _populate_equations(self, model: models.core.Model, /) -> NoReturn:
        """
        """
        eids = list(equations.generate_eids_by_kind(model._steady_equations, equations.EquationKind.STEADY_EVALUATOR))
        self._equations = [ eqn for eqn in model._steady_equations if eqn.id in eids ]


    def _populate_quantities(self, model: models.core.Model, /) -> NoReturn:
        """
        """
        qids = list(quantities.generate_qids_by_kind(model._quantities, quantities.QuantityKind.STEADY_EVALUATOR))
        self._quantities = [ qty for qty in model._quantities if qty.id in qids ]


    def _create_evaluator_function(self, /) -> NoReturn:
        """
        """
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t={self._t_zero}: numpy.array([{func_string}], dtype=float)")


    def _create_incidence_matrix(self, /) -> NoReturn:
        """
        """
        matrix = numpy.zeros((self.num_equations, self.num_quantities), dtype=int)
        qid_to_column = { qid: column for column, qid in enumerate(self._qids) }
        for row_index, eqn in enumerate(self._equations):
            column_indices = list(set(
                qid_to_column[tok.qid]
                for tok in eqn.incidence if tok.qid in self._qids
            ))
            matrix[row_index, column_indices] = 1
        self.incidence_matrix = matrix


    def _z_from_x(self, /):
        return self._x[self._qids, self._t_zero]


    def _x_from_z(self, z: numpy.ndarray, /) -> numpy.ndarray:
        x = numpy.copy(self._x)
        if z is not None:
            x[self._qids, :] = numpy.reshape(z, (-1,1))
        return x
#]

