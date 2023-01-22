"""
"""


#[
from __future__ import annotations

import numpy
import math

from typing import NoReturn
from numbers import Number
from collections.abc import Iterable, Sequence

from .exceptions import ListException
from .incidence import (
    Token, Tokens, get_max_shift, 
    get_min_shift, get_max_qid,
)
from .equations import (
    X_REF_PATTERN, EquationKind, Equation, Equations,
    generate_all_tokens_from_equations, 
    create_evaluator_func_string,
)
#]


class ValueMixin:
    #[
    @property
    def value(self):
        return self._value if self._value is not None else self._data_context[self._data_index]
    #]


class LoglyMixin:
    #[
    @property
    def logly(self):
        return self._logly if self._logly is not None else self._logly_context.get(self._logly_index, False)
    #]


class DiffernAtom(ValueMixin, LoglyMixin):
    """
    Atomic value for differentiation
    """
    _data_context: numpy.ndarray | None = None
    _logly_context: dict[int, bool] | None = None
    _is_atom: bool = True
    #[
    def __init__(self) -> NoReturn:
        """
        """
        self._value = None
        self._diff = None
        self._logly = None
        self._data_index = None
        self._logly_index = None


    @classmethod
    def no_context(
        cls: type,
        value: Number,
        diff: Number,
        logly: bool,
    ) -> Self:
        """
        Create atom with self-contained data and logly contexts
        """
        self = cls()
        self._value = value
        self._diff = diff
        self._logly = logly
        return self


    @classmethod
    def in_context(
            cls: type,
            diff: numpy.ndarray,
            token: Token, 
            columns_to_eval: tuple[int, int],
        ):
        """
        Create atom with pointers to data and logly contexts
        """
        self = cls()
        self._diff = diff if numpy.any(diff!=0) else 0
        self._data_index = (
            slice(token.qid, token.qid+1),
            slice(columns_to_eval[0]+token.shift, columns_to_eval[1]+token.shift+1),
        )
        self._logly_index = token.qid
        return self


    @classmethod
    def zero_atom(
        cls: type,
        diff_shape: tuple[int, int],
    ) -> Self:
        return DiffernAtom.no_context(0, numpy.zeros(diff_shape), False)


    @property
    def diff(self):
        return self._diff if not self.logly else self._diff * self.value


    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return DiffernAtom.no_context(new_value, new_diff, False)


    def __add__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return DiffernAtom.no_context(new_value, new_diff, False)


    def __sub__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        new_logly = False
        return DiffernAtom.no_context(new_value, new_diff, False)


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
        return DiffernAtom.no_context(new_value, new_diff, False)


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
        return DiffernAtom.no_context(new_value, new_diff, False)


    def __rtruediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        new_value = other / self_value
        new_diff = -other*self_diff / (self_value**2)
        return DiffernAtom.no_context(new_value, new_diff, False)


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
        return DiffernAtom.no_context(new_value, new_diff, False)


    def _exponential(self, other):
        """
        Differenatiate exponential function other**self(x)
        """
        new_value = other**self.value
        new_diff = other**self.value * numpy.log(other) * self.diff
        return new_value, new_diff


    def _power(self, other):
        """
        Differenatiate power function self(x)**other
        """
        self_value = self.value
        self_diff = self.diff
        new_value = self_value ** other
        new_diff = other * (self_value**(other-1)) * self_diff
        return new_value, new_diff


    __rmul__ = __mul__
    __radd__ = __add__


    def __rsub__(self, other):
        """
        Differenatiate other - self
        """
        return self.__neg__().__add__(other)


    def log(self):
        new_value = numpy.log(self.value)
        new_diff = 1 / self.value * self.diff
        return DiffernAtom.no_context(new_value, new_diff, False)


    def exp(self):
        new_value = numpy.exp(self.value)
        new_diff = new_value * self.diff
        return DiffernAtom.no_context(new_value, new_diff, False)
    #]


class InvarianceAtom(LoglyMixin):
    """
    Atomic value for invariance testing
    """
    _data_context: numpy.ndarray | None = None
    _logly_context: dict[int, bool] | None = None
    _is_atom: bool = True
    #[
    def __init__(self) -> NoReturn:
        """
        """
        self._diff = None
        self._invariant = None
        self._logly = None
        self._logly_index = None


    @classmethod
    def no_context(
        cls: type,
        diff: numpy.ndarray,
        invariant: numpy.ndarray,
        logly: bool,
    ) -> Self:
        self = cls()
        self._diff = diff
        self._invariant = invariant
        return self


    @classmethod
    def in_context(
        cls: type,
        diff: numpy.ndarray,
        token: Token,
        *args,
    ) -> Self:
        """
        """
        self = cls()
        self._diff = diff==1
        self._invariant = numpy.full((diff.shape[0], 1), True)
        self._logly_index = token.qid
        return self


    def __add__(self, other: Self | Number) -> Self:
        """
        Invariance of self(x)+other(x) or self(x)+other
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
            new_invariant = self._invariant & other._invariant
        else:
            new_diff = self._diff
            new_invariant = self._invariant
        return InvarianceAtom.no_context(new_diff, new_invariant, False)


    def __mul__(self, other: Self | Number):
        """
        Invariance of self(x)*other(x) or self(x)*other
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
            if numpy.all(self._diff==False):
                new_invariant = other._invariant
                new_logly = other.logly
            elif numpy.all(other._diff==False):
                new_invariant = self._invariant
                new_logly = self.logly
            else:
                new_invariant = (
                    numpy.logical_not(self._diff) & other._invariant
                    & numpy.logical_not(other._diff) & self._invariant
                )
                new_logly = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_logly = self.logly
        return InvarianceAtom.no_context(new_diff, new_invariant, new_logly)


    def __rmul__(self, other):
        """
        Invariance of other*self(x)
        """
        return InvarianceAtom.no_context(self._diff, self._invariant, self._logly)


    def __truediv__(self, other):
        """
        Invariance of self(x)/other(x) or self(x)/other
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
            new_invariant = numpy.all(other._diff, axis=0) & self._invariant
            new_logly = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_logly = self._logly
        return InvarianceAtom.no_context(new_diff, new_invariant, new_logly)


    def __rtruediv__(self, other: Number) -> Self:
        """
        Invariance of self(x) / other
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom.no_context(new_diff, new_invariant, False)


    __sub__ = __add__
    __radd__ = __add__
    __rsub__ = __add__


    def log(self) -> Self:
        """
        Invariance of log(self(x))
        """
        new_diff = self._diff
        new_invariant = new_invariant if self._logly else (False & self._invariant)
        return InvarianceAtom.no_context(new_diff, new_invariant, False)


    def _unary(self) -> Self:
        """
        Invariance of f(self(x))
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom.no_context(new_diff, new_invariant, False)


    def _binary(self, other: Self | Number) -> Self:
        """
        Invariance of f(self(x), other(x)) or f(self(x), other)
        """
        if hasattr(other, "_is_atom"):
            new_diff = self._diff | other._diff
        else:
            new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom.no_context(new_diff, new_invariant, False)


    exp = _unary
    __pow__ = _binary
    #]


def log(x): 
    """
    """
    #[
    if isinstance(x, Number):
        return math.log(x)
    else:
        return x.log()
    #]


def exp(x): 
    """
    """
    #[
    if isinstance(x, Number):
        return math.exp(x)
    else:
        return x.exp()
    #]


class Context:
    """
    """
    #[
    def __init__(
        self,
        atom_class: type,
    ) -> NoReturn:
        """
        Initialize Audi contextual space
        """
        self._atom_class = atom_class
        self._x = None
        self._func_string = None
        self._func = None


    @classmethod
    def for_equations(
        cls: type,
        atom_class: type,
        wrt_tokens: Tokens,
        eids: Iterable[int],
        source_equations: Equations,
        num_columns_to_eval: int = 1,
        /,
    ) -> Self:
        """
        """
        self = cls(atom_class)

        equations = sort_equations(eqn for eqn in source_equations if eqn.id in eids)

        all_tokens = set(generate_all_tokens_from_equations(equations))
        min_shift = get_min_shift(all_tokens)
        max_shift = get_max_shift(all_tokens)
        self.shape_data = (
            1 + (get_max_qid(all_tokens) or 0),
            -min_shift + num_columns_to_eval + max_shift,
        )

        t_zero = -min_shift
        self._columns_to_eval = (t_zero, t_zero + num_columns_to_eval - 1)

        eid_to_wrt_tokens = create_eid_to_wrt_tokens(equations, wrt_tokens)
        self._populate_atom_array(equations, eid_to_wrt_tokens, self._columns_to_eval)

        xtrings = [ 
            _create_audi_xtring(eqn, eid_to_wrt_tokens[eqn.id]) 
           for eqn in equations 
        ]

        self._func_string = create_evaluator_func_string(xtrings)
        self._func = eval(self._func_string)

        return self


    def _populate_atom_array(
        self,
        equations: Equations,
        eid_to_wrt_tokens: dict[int, Tokens],
        columns_to_eval: tuple[int, int],
    ) -> NoReturn:
        """
        """
        x = {}
        atom_constructor_in_context = self._atom_class.in_context
        for eqn in equations:
            wrt_tokens_here = eid_to_wrt_tokens[eqn.id]
            for tok in eqn.incidence:
                key = _create_audi_key(tok, eqn.id)
                diff = _diff_value_for_atom_from_incidence(tok, wrt_tokens_here)
                x[key] = atom_constructor_in_context(diff, tok, columns_to_eval)
        self._x = x


    def eval(
        self,
        data_context: numpy.ndarray,
        logly_context: dict[int, bool],
    ) -> Iterable[Atom]:
        self._verify_data_array_shape(data_context.shape)
        self._atom_class._data_context = data_context
        self._atom_class._logly_context = logly_context
        output = self._func(self._x, None)
        self._atom_class._data_context = None
        self._atom_class._logly_context = None
        return output


    def _verify_data_array_shape(self, shape_data: numpy.ndarray) -> NoReturn:
        """
        """
        if shape_data[0]>=self.shape_data[0] and shape_data[1]>=self.shape_data[1]:
            return
        raise InvalidInputDataArrayShape(shape_data, self.shape_data)
    #]


def _diff_value_for_atom_from_incidence(
    token: Token,
    wrt_tokens: Tokens,
) -> numpy.ndarray:
    """
    """
    #[
    diff = numpy.zeros((len(wrt_tokens), 1))
    if token in wrt_tokens:
        diff[wrt_tokens.index(token)] = 1
    return diff
    #]


def _create_audi_xtring(
    equation: Equation,
    wrt_tokens: Tokens,
) -> str:
    """
    """
    #[
    xtring = equation.replace_equation_ref_in_xtring(equation.id)
    xtring = xtring.replace("[", "['").replace("]", "']")
    return xtring
    #]


def _create_audi_key(
    token: Token,
    eid: int,
) -> str:
    """
    Craete a hashable representation of a token in an Audi expression
    """
    #[
    return X_REF_PATTERN.format( 
        qid=token.qid,
        shift=token.shift,
        eid=eid,
    )
    #]


class InvalidInputDataArrayShape(ListException):
    """
    """
    #[
    def __init__(
        self,
        shape_entered: tuple[int, int],
        needed: tuple[int, int]
    ) -> NoReturn:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)
    #]

