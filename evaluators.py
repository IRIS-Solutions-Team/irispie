"""
"""

#[
from __future__ import annotations
from IPython import embed
import enum as en_
import numpy as np_
import copy as co_
from numpy import (log, exp, )
from typing import (Self, NoReturn, )
from collections.abc import (Iterable, )

from .models import core
from .models import variants
from . import quantities
from . import equations
from .functions import *
from . import equations
#]


class EvaluatorKind(en_.Flag, ):
    """
    """
    #[
    PLAIN = en_.auto()
    STEADY = en_.auto()
    STACKED = en_.auto()
    PERIOD = en_.auto()
    #]


class SteadyEvaluator:
    """
    """
    #[
    kind = EvaluatorKind.STEADY

    @classmethod
    def for_model(
        cls,
        in_model: core.Model,
        in_equations: equations.Equations,
        /,
    ) -> Self:
        self = cls()
        self._equations = []
        self._quantities = []
        self._populate_equations(in_equations)
        self._populate_quantities(in_model)
        self._qids = list(quantities.generate_all_qids(self._quantities))
        self.min_shift = equations.get_min_shift_from_equations(self._equations)
        self.max_shift = equations.get_max_shift_from_equations(self._equations)
        self._create_evaluator_function()
        self._create_incidence_matrix()
        self._x = None
        self._L = None
        self._z = None
        return self

    @property
    def init(self, /, ) -> np_.ndarray:
        return np_.copy(self._z0)

    @property
    def quantities_human(self, /, ) -> Iterable[str]:
        return [ qty.human for qty in self._quantities ]

    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self, /, ) -> int:
        return len(self._equations)

    @property
    def num_quantities(self, /, ) -> int:
        return len(self._quantities)

    @property
    def t_zero(self, /, ) -> int:
        return -self.min_shift

    @property
    def num_columns(self, /, ) -> int:
        return -self.min_shift + 1 + self.max_shift

    def update_steady_array(self, model, variant, /, ) -> Self:
        """
        """
        self._x = model.create_steady_array(variant, num_columns=self.num_columns, shift_in_first_column=self.min_shift)
        self._z0 = self._z_from_x()
        self._L = co_.deepcopy(self._x)
        return self

    def eval(self, current: np_.ndarray | None = None, /, ):
        """
        """
        x = self._x_from_z(current)
        return self._func(x, self.t_zero, self._L)

    def _populate_equations(self, in_equations: equations.Equations, /, ) -> NoReturn:
        """
        """
        eids = list(equations.generate_eids_by_kind(in_equations, equations.EquationKind.STEADY_EVALUATOR))
        self._equations = [ eqn for eqn in in_equations if eqn.id in eids ]

    def _populate_quantities(self, model: core.Model, /, ) -> NoReturn:
        """
        """
        qids = list(quantities.generate_qids_by_kind(model._quantities, quantities.QuantityKind.STEADY_EVALUATOR))
        self._quantities = [ qty for qty in model._quantities if qty.id in qids ]

    def _create_evaluator_function(self, /, ) -> NoReturn:
        """
        """
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t, L: np_.array([{func_string}], dtype=float)")

    def _create_incidence_matrix(self, /, ) -> NoReturn:
        """
        """
        matrix = np_.zeros((self.num_equations, self.num_quantities), dtype=int)
        qid_to_column = { qid: column for column, qid in enumerate(self._qids) }
        for row_index, eqn in enumerate(self._equations):
            column_indices = list(set(
                qid_to_column[tok.qid]
                for tok in eqn.incidence if tok.qid in self._qids
            ))
            matrix[row_index, column_indices] = 1
        self.incidence_matrix = matrix


    def _z_from_x(self, /, ):
        return self._x[self._qids, self.t_zero]


    def _x_from_z(self, z: np_.ndarray, /, ) -> np_.ndarray:
        x = np_.copy(self._x)
        if z is not None:
            x[self._qids, :] = np_.reshape(z, (-1,1))
        return x
#]


class PlainEvaluator:
    """
    """
    #[
    kind = EvaluatorKind.PLAIN

    @classmethod
    def for_model(
        cls,
        in_model: core.Model,
        in_equations: equations.Equations,
        /,
    ) -> Self:
        self = cls()
        self._equations = []
        self._populate_equations(in_equations)
        self.min_shift = equations.get_min_shift_from_equations(self._equations)
        self.max_shift = equations.get_max_shift_from_equations(self._equations)
        self._create_evaluator_function()
        return self

    @property
    def equations_human(self, /, ) -> Iterable[str]:
        return [ eqn.human for eqn in self._equations ]

    @property
    def num_equations(self, /, ) -> int:
        return len(self._equations)

    def eval(self, dataslab: np_.ndarray, column_slice: slice, /, ) -> np_.ndarray:
        """
        """
        L = None
        return self._func(dataslab, column_slice, L, )

    def _populate_equations(self, in_equations: equations.Equations, /, ) -> NoReturn:
        """
        """
        eids = list(equations.generate_eids_by_kind(in_equations, equations.EquationKind.PLAIN_EVALUATOR))
        self._equations = [ eqn for eqn in in_equations if eqn.id in eids ]

    def _create_evaluator_function(self, /, ) -> NoReturn:
        """
        """
        xtrings = [ eqn.remove_equation_ref_from_xtring() for eqn in self._equations ]
        func_string = ",".join(xtrings)
        self._func = eval(f"lambda x, t, L: np_.array([{func_string}], dtype=float)")
#]

