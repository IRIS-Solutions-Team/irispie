"""
"""


#[
import functools as _ft
import numpy as _np
import itertools as _it
from numbers import (Number, )
from collections.abc import (Iterable, Callable, )
from typing import (TypeAlias, Any, Literal, )

from .. import (equations as _eq, quantities as _qu, wrongdoings as _wd, )
from . import (variants as _va, flags as _mg, )
from ..fords import (steadiers as _fs, )
from ..evaluators import (steady as _es, )
#]


_SteadySolverReturn: TypeAlias = tuple[
    _np.ndarray, tuple[int], _np.ndarray, tuple[int],
]

_EquationSwitch: TypeAlias = Literal["steady"] | Literal["dynamic"]


class SteadyMixin:
    """
    """
    def steady(
        self,
        /,
        **kwargs, 
    ) -> list[dict]:
        """
        Calculate steady state for each Variant within this Model
        """
        model_flags = _mg.Flags.update_from_kwargs(self.get_flags(), **kwargs)
        solver = self._choose_steady_solver(model_flags.is_linear, model_flags.is_flat, )
        info = tuple({} for _ in self._variants)
        for i, v in enumerate(self._variants):
            levels, qids_levels, changes, qids_changes = solver(v, model_flags, )
            v.update_levels_from_array(levels, qids_levels, )
            v.update_changes_from_array(changes, qids_changes, )
        return info


    def _steady_linear(
        self, 
        variant: _va.Variant,
        model_flags: _mg.Flags,
        /,
        algorithm: Callable,
    ) -> _SteadySolverReturn:
        """
        """
        #
        # Calculate first-order system for steady equations for this variant
        system = self._systemize(
            variant,
            self._invariant._steady_descriptor,
            model_flags,
        )
        #
        #=======================================================================
        # Core algorithm: Calculate steady state for this variant
        Xi, Y, dXi, dY = algorithm(system)
        levels = _np.hstack(( Xi.flat, Y.flat )).flatten()
        changes = _np.hstack(( dXi.flat, dY.flat )).flatten()
        #=======================================================================
        #
        # Extract only tokens with zero shift
        tokens = tuple(_it.chain(
            self._invariant._steady_descriptor.system_vectors.transition_variables,
            self._invariant._steady_descriptor.system_vectors.measurement_variables,
        ))
        #
        # True for tokens with zero shift, e.g. [True, False, True, ... ]
        zero_shift_index = [ not t.shift for t in tokens ]
        #
        # Extract steady levels for quantities with zero shift
        levels = levels[zero_shift_index]
        changes = changes[zero_shift_index]
        qids = tuple(t.qid for t in _it.compress(tokens, zero_shift_index))
        #
        # Delogarithmize when needed
        qid_to_logly = self.create_qid_to_logly()
        levels = _apply_delog_on_vector(levels, qids, qid_to_logly)
        changes = _apply_delog_on_vector(changes, qids, qid_to_logly)
        #
        return levels, qids, changes, qids

    _steady_linear_flat = _ft.partialmethod(_steady_linear, algorithm=_fs.solve_steady_linear_flat)
    _steady_linear_nonflat = _ft.partialmethod(_steady_linear, algorithm=_fs.solve_steady_linear_nonflat)


    def _steady_nonlinear_flat(
        self,
        variant: _va.Variant,
        /,
    ) -> _SteadySolverReturn:
        """
        """
        return None, None, None, None


    def _steady_nonlinear_nonflat(
        self,
        variant: _va.Variant,
        /,
    ) -> _SteadySolverReturn:
        """
        """
        return None, None, None, None


    def _choose_steady_solver(
        self,
        is_linear: bool,
        is_flat: bool,
        /,
    ) -> Callable:
        """
        Choose steady solver depending on linear and flat flags
        """
        match (is_linear, is_flat):
            case (False, False):
                return self._steady_nonlinear_nonflat
            case (False, True):
                return self._steady_nonlinear_flat
            case (True, False):
                return self._steady_linear_nonflat
            case (True, True):
                return self._steady_linear_flat


    def check_steady(
        self,
        /,
        equation_switch: _EquationSwitch = "dynamic",
        when_fails: _wd.HOW = "error",
        tolerance: float = 1e-12,
    ) -> tuple[bool, tuple[dict, ...]]:
        """
        Verify currently assigned steady state in dynamic or steady equations for each Variant within this Model
        """
        qid_to_logly = self.create_qid_to_logly()
        equator = self._choose_plain_equator(equation_switch)
        steady_arrays = (
            v.create_steady_array(
                qid_to_logly,
                num_columns=equator.min_num_columns + 1,
                shift_in_first_column=equator.min_shift,
            ) for v in self._variants
        )
        # REFACTOR
        t_zero = -equator.min_shift
        dis = [ 
            _np.hstack((
                equator.eval(x, t_zero, x[:, t_zero]),
                equator.eval(x, t_zero+1, x[:, t_zero+1]),
            ))
            for x in steady_arrays
        ]
        # REFACTOR
        max_abs_dis = [ _np.max(_np.abs(d)) for d in dis ]
        status = [ d < tolerance for d in max_abs_dis ]
        all_status = all(status)
        if not all_status:
            message = "Invalid steady state"
            _wd.throw(when_fails, message)
        details = tuple(
            {"discrepancies": d, "max_abs_discrepancy": m, "is_valid": s}
            for d, m, s in zip(dis, max_abs_dis, status)
        )
        return all_status, details


def _apply_delog_on_vector(
    vector: _np.ndarray,
    qids: Iterable[int],
    qid_to_logly: dict[int, bool],
    /,
) -> _np.ndarray:
    """
    Delogarithmize the elements of numpy vector that have True log-status
    """
    logly_index = [ qid_to_logly[qid] for qid in qids ]
    vector = _np.copy(vector)
    if any(logly_index):
        vector[logly_index] = _np.exp(vector[logly_index])
    return vector

