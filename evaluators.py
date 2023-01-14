"""
"""

#[
from __future__ import annotations

import enum
import numpy

from typing import Self, NoReturn
from collections.abc import Iterable

from .models import (
    Model,
)
#]


class EvaluatorKind(enum.Flag, /):
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

    def __init__(self, model: Model, /) -> NoReturn:
        self._equations = []
        self._quantities = []
        self._populate_equations(model)
        self._populate_quantities(model)
        self._qids = list(generate_all_qids(model._quantities))
        self._t_zero = -model._get_min_shift()
        self._create_evaluator_function()
        self._create_incidence_matrix()
        num_columns = self._t_zero + 1 + model._get_max_shift()
        self._x: numpy.ndarray = model.create_steady_array(num_columns)
        self._z0: numpy.ndarray = self._z_from_x()


    @property
    def init(self, /) -> numpy.ndarray:
        return numpy.copy(self._z0)


    @property
    def quantities_solved(self, /) -> Iterable[str]:
        return [ qty.human for qty in self._quantities ]


    @property
    def equations_solved(self, /) -> Iterable[str]:
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


    def _populate_equations(self, model: Model, /) -> NoReturn:
        """
        """
        eids = list(generate_eids_by_kind(model._equations, EquationKind.EQUATION))
        self._equations = [ eqn for eqn in model._equations if eqn.id in eids ]


    def _populate_quantities(self, model: Model, /) -> NoReturn:
        """
        """
        qids = list(generate_qids_by_kind(model._quantities, QuantityKind.VARIABLE))
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
        for row_index, eqn in enumerate(self._equations):
            column_indices = list(set( tok.qid for tok in eqn.incidence if tok.qid in self._qids ))
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

