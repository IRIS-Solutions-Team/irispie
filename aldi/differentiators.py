"""
Algorithmic differentiator
"""


#[
from __future__ import annotations
# from IPython import embed

from typing import (Self, NoReturn, )
from numbers import (Number, )
from collections.abc import Iterable, Sequence

import numpy as np_

from ..exceptions import ListException

from ..incidence import (
    Token, Tokens, get_max_shift, 
    get_min_shift, get_max_qid,
)

from ..equations import (
    X_REF_PATTERN, EquationKind, Equation, Equations,
    generate_all_tokens_from_equations, 
    create_evaluator_func_string,
)

from ..functions import *
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


class Atom(ValueMixin, LoglyMixin):
    """
    Atomic value for differentiation
    """
    _data_context: np_.ndarray | None = None
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
            diff: np_.ndarray,
            token: Token, 
            columns_to_eval: tuple[int, int],
        ):
        """
        Create atom with pointers to data and logly contexts
        """
        self = cls()
        self._diff = diff if np_.any(diff!=0) else 0
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
        return Atom.no_context(0, np_.zeros(diff_shape), False)

    @property
    def diff(self):
        return self._diff if not self.logly else self._diff * self.value

    def __pos__(self):
        return self

    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return Atom.no_context(new_value, new_diff, False)

    def __add__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return Atom.no_context(new_value, new_diff, False)

    def __sub__(self, other):
        if hasattr(other, "_is_atom"):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        new_logly = False
        return Atom.no_context(new_value, new_diff, False)

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
        return Atom.no_context(new_value, new_diff, False)

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
        return Atom.no_context(new_value, new_diff, False)

    def __rtruediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        new_value = other / self_value
        new_diff = -other*self_diff / (self_value**2)
        return Atom.no_context(new_value, new_diff, False)

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
        return Atom.no_context(new_value, new_diff, False)

    def _exponential(self, other):
        """
        Differenatiate exponential function other**self(x)
        """
        new_value = other**self.value
        new_diff = other**self.value * np_.log(other) * self.diff
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

    def _log_(self):
        new_value = np_.log(self.value)
        new_diff = 1 / self.value * self.diff
        return Atom.no_context(new_value, new_diff, False)

    def _exp_(self):
        new_value = np_.exp(self.value)
        new_diff = new_value * self.diff
        return Atom.no_context(new_value, new_diff, False)
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
        equations: Equations,
        eid_to_wrt_tokens: dict[int, Tokens],
        num_columns_to_eval: int = 1,
        /,
    ) -> Self:
        """
        """
        self = cls(atom_class)

        equations = list(equations)
        all_tokens = set(generate_all_tokens_from_equations(equations))
        min_shift = get_min_shift(all_tokens)
        max_shift = get_max_shift(all_tokens)
        self.shape_data = (
            1 + (get_max_qid(all_tokens) or 0),
            -min_shift + num_columns_to_eval + max_shift,
        )

        t_zero = -min_shift
        self._columns_to_eval = (t_zero, t_zero + num_columns_to_eval - 1)

        self._populate_atom_array(
            equations,
            eid_to_wrt_tokens,
            self._columns_to_eval,
        )

        xtrings = [ 
            _create_aldi_xtring(eqn, eid_to_wrt_tokens[eqn.id]) 
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
                key = _create_aldi_key(tok, eqn.id)
                diff = _diff_value_for_atom_from_incidence(tok, wrt_tokens_here)
                x[key] = atom_constructor_in_context(diff, tok, columns_to_eval)
        self._x = x


    def eval(
        self,
        data_context: np_.ndarray,
        logly_context: dict[int, bool],
    ) -> Iterable[Atom]:
        """
        Evaluate and return a list of final atoms, one for each equation
        """
        self._verify_data_array_shape(data_context.shape)
        self._atom_class._data_context = data_context
        self._atom_class._logly_context = logly_context
        output = self._func(self._x, None)
        self._atom_class._data_context = None
        self._atom_class._logly_context = None
        return output

    def eval_to_arrays(
        self,
        *args,
    ) -> tuple[np_.ndarray, np_.ndarray]:
        """
        Evaluate and return arrays of diffs and values extracted from final atoms
        """
        output = self.eval(*args)
        return (
            np_.vstack([x.diff for x in output]),
            np_.vstack([x.value for x in output]),
        )

    def _verify_data_array_shape(self, shape_data: np_.ndarray) -> NoReturn:
        """
        """
        if shape_data[0]>=self.shape_data[0] and shape_data[1]>=self.shape_data[1]:
            return
        raise InvalidInputDataArrayShape(shape_data, self.shape_data)
    #]


def _diff_value_for_atom_from_incidence(
    token: Token,
    wrt_tokens: Tokens,
) -> np_.ndarray:
    """
    """
    #[
    diff = np_.zeros((len(wrt_tokens), 1))
    if token in wrt_tokens:
        diff[wrt_tokens.index(token)] = 1
    return diff
    #]


def _create_aldi_xtring(
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


def _create_aldi_key(
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


