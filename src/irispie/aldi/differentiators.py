"""
Algorithmic differentiator
"""


#[
from __future__ import annotations

from typing import (Self, Callable, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, Sequence, )
import numpy as _np
import copy as _cp

from ..exceptions import ListException

from ..aldi import finite_differentiators as af_
from ..aldi import adaptations as _adaptations
from ..incidences import main as _incidences
from ..equators import plain as _equators
from .. import equations as _equations
#]


class ValueMixin:
    #[
    @property
    def value(self):
        return self._value if self._value is not None else type(self)._data_context[self._data_index]
    #]


class Atom(ValueMixin, ):
    """
    Atomic value for differentiation
    """
    _data_context: _np.ndarray | None = None
    _is_atom: bool = True
    #[
    def __init__(self) -> None:
        """
        """
        self._value = None
        self._diff = None
        self._logly = None
        self._data_index = None

    @classmethod
    def no_context(
        cls,
        value: Number,
        diff: Number,
        /,
        logly: bool | None = False,
    ) -> Self:
        """
        Create atom with self-contained data and logly contexts
        """
        self = cls()
        self._value = value
        self._diff = diff
        self._logly = logly if logly is not None else False
        return self

    @classmethod
    def in_context(
        cls,
        /,
        diff: _np.ndarray | Number,
        data_index: tuple[int, slice],
        logly: bool,
    ) -> Self:
        """
        Create atom with pointers to data and logly contexts
        """
        self = Atom()
        self._diff = diff
        self._data_index = data_index
        self._logly = logly if logly is not None else False
        return self

    @classmethod
    def zero(
        cls,
        diff_shape: tuple[int, int],
    ) -> Self:
        return cls.no_context(0, _np.zeros(diff_shape), False)

    @property
    def diff(self):
        return self._diff if not self._logly else self._diff * self.value

    def __pos__(self):
        return self

    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return type(self).no_context(new_value, new_diff, False)

    def __add__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return type(self).no_context(new_value, new_diff, False)

    def __sub__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        new_logly = False
        return type(self).no_context(new_value, new_diff, False)

    def __mul__(self, other):
        self_value = self.value
        self_diff = self.diff
        if hasattr(other, "_is_atom"):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value * other_value
            new_diff = self_diff * other_value + self_value * other_diff
        else:
            new_value = self_value * other
            new_diff = self_diff * other
        return type(self).no_context(new_value, new_diff, False)

    def __truediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        if hasattr(other, "_is_atom"):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value / other_value
            new_diff = (self_diff*other_value - self_value*other_diff) / (other_value**2)
        else:
            new_value = self_value / other
            new_diff = self_diff / other
        return type(self).no_context(new_value, new_diff, False)

    def __rtruediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        new_value = other / self_value
        new_diff = -other*self_diff / (self_value**2)
        return type(self).no_context(new_value, new_diff, False)

    def __pow__(self, other):
        """
        Differenatiate self**other 
        """
        if hasattr(other, "_is_atom"):
            # self(x)**other(x)
            # d[self(x)**other(x)] = d[self(x)**other] + d[self**other(x)]
            new_value = self.value**other.value
            _, new_diff_exp = self._power(other.value) # self(x)**other
            _, new_diff_pow = other._exponential(self.value) # self**other(x)
            new_diff = new_diff_exp + new_diff_pow
        else:
            # self(x)**other
            new_value, new_diff = self._power(other)
        return type(self).no_context(new_value, new_diff, False)

    def _exponential(self, other_value):
        """
        Differenatiate exponential function other_value**self(x)
        """
        new_value = other_value**self.value
        new_diff = (
            other_value**self.value * _np.log(other_value) * self.diff 
            if other_value != 0 else 0
        )
        return new_value, new_diff

    def _power(self, other_value):
        """
        Differenatiate power function self(x)**other_value
        """
        self_value = self.value
        self_diff = self.diff
        new_value = self_value ** other_value
        new_diff = other_value * (self_value**(other_value-1)) * self_diff
        return new_value, new_diff

    __rmul__ = __mul__

    __radd__ = __add__

    def __rsub__(self, other, /, ) -> Self:
        """
        Differenatiate other - self
        """
        return self.__neg__().__add__(other)

    def _log_(self, /, ) -> Self:
        """
        Differentiate log(self)
        """
        new_value = _np.log(self.value)
        new_diff = 1 / self.value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def _exp_(self, /, ) -> Self:
        """
        Differentiate exp(self)
        """
        new_value = _np.exp(self.value)
        new_diff = new_value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def _sqrt_(self, /, ) -> Self:
        """
        Differentiate sqrt(self)
        """
        new_value = _np.sqrt(self.value)
        new_diff = 0.5 / _np.sqrt(self.diff)
        return type(self).no_context(new_value, new_diff, False)

    def _maximum_(
        self,
        floor: Number = 0,
        /, 
    ) -> Self:
        """
        Differenatiate maximum(self, floor)
        """
        orig_value = self.value
        orig_diff = self.diff
        if isinstance(orig_value, Number) and isinstance(orig_diff, Number):
            orig_value = _np.array(orig_value, dtype=float)
            orig_diff = _np.array(orig_diff, dtype=float)
        new_value = _np.copy(orig_value)
        inx_floor = orig_value == floor
        inx_below = orig_value < floor
        inx_above = orig_value > floor
        new_value[inx_below] = floor
        new_diff = _np.copy(orig_diff)
        multiplier = _np.copy(orig_value)
        multiplier[inx_floor] = 0.5
        multiplier[inx_above] = 1
        multiplier[inx_below] = 0
        new_diff = orig_diff * multiplier
        return type(self).no_context(new_value, new_diff, False)

    def _mininum_(
        self,
        ceiling: Number = 0,
        /,
    ) -> Self:
        """
        Differenatiate minimum(self, ceiling)
        """
        return (-self)._max_(-ceiling)
    #]


class Context:
    """
    """
    #[
    def __init__(
        self,
        equations: Iterable[_equations.Equation],
        atom_factory: AtomFactoryProtocol,
        /,
        eid_to_wrts: dict[int, tuple[Any]],
        qid_to_logly: dict[int, bool] | None,
        num_columns_to_eval: int,
        custom_functions: dict | None,
    ) -> Self:
        """
        """
        self._eid_to_wrts = eid_to_wrts
        self._qid_to_logly = qid_to_logly or {}
        self._num_columns_to_eval = num_columns_to_eval
        self._populate_equations(equations, )
        #
        all_tokens = set(_equations.generate_all_tokens_from_equations(self._equations, ), )
        self.min_shift = _incidences.get_min_shift(all_tokens, )
        self.max_shift = _incidences.get_max_shift(all_tokens, )
        self.shape_data = (
            1 + (_incidences.get_max_qid(all_tokens) or 0),
            -self.min_shift + num_columns_to_eval + self.max_shift,
        )
        #
        self._populate_atom_dict(atom_factory, )
        #
        custom_functions = { 
            k: af_.finite_differentiator(v) 
            for k, v in custom_functions.items()
        } if custom_functions else None
        custom_functions = _adaptations.add_function_adaptations_to_custom_functions(custom_functions)
        custom_functions["Atom"] = Atom
        #
        self._equator = _equators.PlainEquator(
            self._equations,
            custom_functions=custom_functions,
        )

    @property
    def _t_zero(self, /, ) -> int:
        """
        """
        return -self.min_shift

    def _get_num_wrts(self, eid, /, ) -> int:
        """
        """
        return len(self._eid_to_wrts[eid])

    def _get_columns_to_eval(self, /, ) -> tuple[int, int]:
        """
        """
        return self._t_zero, self._t_zero + self._num_columns_to_eval - 1

    def _get_diff_shape_for_eid(self, eid, /, ) -> tuple[int, int]:
        """
        """
        return self._get_num_wrts(eid), self._num_columns_to_eval

    def _populate_equations(
        self,
        equations: Iterable[_equations.Equation],
        /,
    ) -> None:
        """
        """
        self._equations = tuple(
            _adapt_equation_for_aldi(e, self._get_diff_shape_for_eid(e.id), )
            for e in equations
        )

    def _populate_atom_dict(
        self,
        atom_factory: AtomFactoryProtocol,
        /,
    ) -> None:
        """
        * atom_factory -- Implement create_diff_for_token, create_data_index_for_token, get_logly_for_token
        * qid_to_logly -- Dictionary of qid to logly
        * columns_to_eval -- Tuple of (first, last) column indices to be evaluated
        """
        columns_to_eval = self._get_columns_to_eval()
        self._x = {}
        for eqn in self._equations:
            self._x[eqn.id] = {}
            for tok in eqn.incidence:
                atom = Atom.in_context(
                    diff=atom_factory.create_diff_for_token(tok, self._eid_to_wrts[eqn.id], ),
                    data_index=atom_factory.create_data_index_for_token(tok, columns_to_eval, ),
                    logly=self._qid_to_logly.get(tok.qid, False, ),
                )
                self._x[eqn.id][(tok.qid, tok.shift)] = atom

    def eval(
        self,
        data_array: _np.ndarray,
        steady_array: _np.ndarray,
    ) -> Iterable[Atom]:
        """
        Evaluate and return the list of final atoms, one atom for each equation
        """
        self._verify_data_array_shape(data_array.shape, )
        Atom._data_context = data_array
        output = self._equator.eval(self._x, 0, steady_array, )
        Atom._data_context = None
        return output

    def eval_to_arrays(
        self,
        *args,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        Evaluate and return arrays of diffs and values extracted from final atoms
        """
        output = self.eval(*args)
        diff = _np.vstack([x.diff for x in output])
        value = _np.vstack([x.value for x in output])
        return diff, value

    def eval_diff_to_array(
        self,
        *args,
    ) -> _np.array:
        """
        Evaluate and return array of diffs
        """
        return _np.vstack([
            x.diff for x in self.eval(*args)
            if hasattr(x, "diff")
        ])

    def _verify_data_array_shape(self, shape_data: _np.ndarray) -> None:
        """
        """
        if shape_data[0]>=self.shape_data[0] and shape_data[1]>=self.shape_data[1]:
            return
        raise InvalidInputDataArrayShape(shape_data, self.shape_data)
    #]


def _adapt_equation_for_aldi(
    equation: _eq.Equation,
    diff_shape: tuple[int, int],
    /,
) -> _equations.Equation:
    """
    """
    #[
    aldi_equation = _cp.deepcopy(equation)
    aldi_equation.xtring = aldi_equation.xtring.replace("x[", f"x[{aldi_equation.id}][")
    aldi_equation.xtring += f" + Atom.zero({diff_shape}, )"
    return aldi_equation
    #]


class InvalidInputDataArrayShape(ListException, ):
    """
    """
    #[
    def __init__(
        self,
        shape_entered: tuple[int, int],
        needed: tuple[int, int]
    ) -> None:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)
    #]


class AtomFactoryProtocol(Protocol, ):
    """
    Protocol for creating atoms representing tokens
    """
    #[
    def create_diff_for_token(
        self,
        token: _incidences.Token,
        wrts: tuple[Any, ...],
    ) -> _np.ndarray:
        """
        Create a diff for an atom representing a given token
        """
        ...

    def create_data_index_for_token(
        self,
        token: _incidences.Token,
        columns_to_eval: tuple[int, int],
    ) -> tuple[int, slice]:
        """
        Create a data index for an atom representing a given token
        """
        ...
    #]


