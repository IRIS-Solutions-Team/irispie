"""
Algorithmic differentiator
"""


#[
from __future__ import annotations

from typing import (Self, Callable, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, Sequence, )
import numpy as _np
import scipy as _sp
import copy as _cp

from ..incidences import main as _incidences
from ..equators import plain as _equators
from .. import equations as _equations
from . import finite_differentiators as af_
from . import adaptations as _adaptations
from . import _rules as _rules

#]


class ValueMixin:
    """
    """
    #[

    #]


class Atom(ValueMixin, ):
    """
    Atomic value for differentiation
    """
    #[

    _data_context: _np.ndarray | None = None
    _column_offset: int | None = None
    _is_atom: bool = True

    __slots__ = (
        '_value',
        '_diff',
        '_logly',
        '_data_index',
        '_row_index',
        '_column_index',
    )

    def __init__(self) -> None:
        """
        """
        self._value = None
        self._diff = None
        self._logly = None
        self._data_index = None
        self._row_index = None
        self._column_index = None

    @classmethod
    def no_context(
        klass,
        value: Number,
        diff: Number,
        /,
        logly: bool | None = False,
    ) -> Self:
        """
        Create atom with self-contained data and logly contexts
        """
        self = klass()
        self._value = value
        self._diff = diff
        self._logly = logly if logly is not None else False
        return self

    @classmethod
    def in_context(
        klass,
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
        self._row_index = data_index[0]
        self._column_index = data_index[1]
        self._logly = logly if logly is not None else False
        return self

    @classmethod
    def zero(
        klass,
        diff_shape: tuple[int, int],
        /,
    ) -> Self:
        return klass.no_context(0, _np.zeros(diff_shape, dtype=_np.float64), False)

    @property
    def value(self, /, ):
        if self._value is not None:
            return self._value
        else:
            column_index = (
                self._column_index + self._column_offset
                if self._column_offset is not None
                else self._column_index
            )
            return self._data_context[self._row_index, column_index]

    @property
    def diff(self):
        return self._diff if not self._logly else self._diff * self.value

    def __pos__(self):
        return self

    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return type(self).no_context(new_value, new_diff, False)

    __add__ = _rules.add_diff

    __sub__ = _rules.sub_diff

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

    def log(self, /, ) -> Self:
        """
        Differentiate log(self)
        """
        new_value = _np.log(self.value)
        new_diff = 1 / self.value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def exp(self, /, ) -> Self:
        """
        Differentiate exp(self)
        """
        new_value = _np.exp(self.value)
        new_diff = new_value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def sqrt(self, /, ) -> Self:
        """
        Differentiate sqrt(self)
        """
        new_value = _np.sqrt(self.value)
        new_diff = 0.5 / _np.sqrt(self.diff)
        return type(self).no_context(new_value, new_diff, False)

    def logistic(self, /, ) -> Self:
        """
        Differentiate logistic(self)
        """
        new_value = _sp.special.expit(self.value, )
        new_diff = new_value * (1 - new_value) * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def maximum(
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

    def mininum(
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
        atom_factory: AtomFactoryProtocol,
        equations: Iterable[_equations.Equation],
        /,
        eid_to_wrts: dict[int, tuple[Any, ...]],
        qid_to_logly: dict[int, bool] | None,
        context: dict | None,
    ) -> Self:
        """
        """
        self._eid_to_wrts = eid_to_wrts
        self._qid_to_logly = qid_to_logly or {}
        self._atom_factory = atom_factory
        self._populate_equations(equations, )
        #
        self._populate_atom_dict()
        #
        context = { 
            k: af_.finite_differentiator(v) 
            for k, v in context.items()
        } if context else None
        context = _adaptations.add_function_adaptations_to_context(context)
        context["Atom"] = Atom
        #
        self._equator = _equators.PlainEquator(
            self._equations,
            context=context,
        )

    def _get_num_wrts(self, eid, /, ) -> int:
        """
        """
        return len(self._eid_to_wrts[eid])

    def _get_diff_shape_for_eid(self, eid, /, ) -> Atom:
        """
        """
        diff = self._atom_factory.create_diff_for_token(None, self._eid_to_wrts[eid], )
        return diff.shape

    def _populate_equations(
        self,
        equations: Iterable[_equations.Equation],
        /,
    ) -> None:
        """
        """
        self._equations = tuple(
            _adapt_equation_for_aldi(e, self._get_diff_shape_for_eid(e.id, ), )
            for e in equations
        )

    def _populate_atom_dict(self, /, ) -> None:
        """
        """
        self._x = {}
        for eqn in self._equations:
            self._x[eqn.id] = {}
            for tok in eqn.incidence:
                atom = Atom.in_context(
                    diff=self._atom_factory.create_diff_for_token(tok, self._eid_to_wrts[eqn.id], ),
                    data_index=self._atom_factory.create_data_index_for_token(tok, ),
                    logly=self._qid_to_logly.get(tok.qid, False, ),
                )
                self._x[eqn.id][(tok.qid, tok.shift)] = atom

    def eval(
        self,
        data_array: _np.ndarray,
        column_offset: int,
        steady_array: _np.ndarray,
    ) -> Iterable[Atom]:
        """
        Evaluate and return the list of final atoms, one atom for each equation
        """
        Atom._data_context = data_array
        Atom._column_offset = column_offset
        output = self._equator.eval(self._x, 0, steady_array, )
        Atom._data_context = None
        Atom._column_offset = None
        return output

    def eval_to_arrays(
        self,
        *args,
        **kwargs,
    ) -> tuple[_np.ndarray, _np.ndarray]:
        """
        Evaluate and return arrays of diffs and values extracted from final atoms
        """
        output = self.eval(*args, **kwargs, )
        diff = _np.vstack([x.diff for x in output])
        value = _np.vstack([x.value for x in output])
        return diff, value

    def eval_diff_to_array(
        self,
        *args,
        **kwargs,
    ) -> _np.array:
        """
        Evaluate and return array of diffs
        """
        return _np.vstack([
            x.diff for x in self.eval(*args, **kwargs, )
            if hasattr(x, "diff")
        ])

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
    ) -> tuple[int, slice]:
        """
        Create a data index for an atom representing a given token
        """
        ...

    #]

