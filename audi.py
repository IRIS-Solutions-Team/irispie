"""
"""


#(
from __future__ import annotations

from numpy import log as np_log, exp as np_exp, zeros, NaN, ndarray, all as np_all
from re import findall, compile
from typing import NamedTuple, Optional, Union, Callable
from numbers import Number
from math import log as math_log, exp as math_exp

from .exceptions import ListException

from .incidence import (
    Token, get_max_shift, 
    get_min_shift, get_max_quantity_id
)

from .equations import (
    X_REF_PATTERN, EquationKind, Equation, create_name_to_id_from_equations,
    collect_all_tokens, create_evaluator_func_string
)
#)




class DiffAtom():
    """
    Atomic value for differentiating equations
    """
#(
    _data: ndarray  = None

    def __init__(
            self, 
            value: Optional[Number]=None,
            diff: Optional[Union[Number, ndarray]]=None,
            data_index: Optional[tuple[slice, slice]]=None,
            log_flag: bool=False,
        ) -> None:
        """
        """
        self._value = value
        self._diff = diff
        self._data_index = data_index
        self._log_flag = log_flag


    @property
    def value(self):
        return self._value if self._value is not None else self._data[self._data_index]


    @property
    def diff(self):
        return self._diff if not self._log_flag else self._diff * self.value


    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return DiffAtom(new_value, new_diff)


    def __add__(self, other):
        if isinstance(other, DiffAtom):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return DiffAtom(new_value, new_diff)


    def __sub__(self, other):
        if isinstance(other, DiffAtom):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        return DiffAtom(new_value, new_diff)


    def __mul__(self, other):
        self_value = self.value
        self_diff = self.diff
        if isinstance(other, DiffAtom):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value * other_value
            new_diff = self_diff * other_value + self_value * other_diff
        else:
            new_value = self_value * other
            new_diff = self_diff * other
        return DiffAtom(new_value, new_diff)


    def __truediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        if isinstance(other, DiffAtom):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value / other_value
            new_diff = (self_diff*other_value - self_value*other_diff) / (other_diff**2)
        else:
            new_value = self_value / other
            new_diff = self_diff / other
        return DiffAtom(new_value, new_diff)


    def __rtruediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        new_value = other / self_value
        new_diff = -other*self_diff / (self_value**2)
        return DiffAtom(new_value, new_diff)


    def __pow__(self, other):
        """
        Differentiate self**other 
        """
        if isinstance(other, DiffAtom):
            # self(x)**other(x)
            # d[self(x)**other(x)] = d[self(x)**other] + d[self**other(x)]
            new_value = self.value**other.value
            _, new_diff_exp = self._power(other.value) # self(x)**other
            _, new_diff_pow = other._exponential(self.value) # self**other(x)
            new_diff = new_diff_exp + new_diff_pow
        else:
            # self(x)**other
            new_value, new_diff = self._power(other)
        return DiffAtom(new_value, new_diff)


    def _exponential(self, other):
        """
        Differentiate exponential function other**self(x)
        """
        new_value = other**self.value
        new_diff = other**self.value * np_log(other) * self.diff
        return new_value, new_diff


    def _power(self, other):
        """
        Differentiate power function self(x)**other
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
        Differentiate other - self
        """
        return self.__neg__().__add__(other)


    def log(self):
        new_value = np_log(self.value)
        new_diff = 1 / self.value * self.diff
        return DiffAtom(new_value, new_diff)


    def exp(self):
        new_value = np_exp(self.value)
        new_diff = new_value * self.diff
        return DiffAtom(new_value, new_diff)
#)





class InvarianceAtom():
    """
    Atomic value for investigating the invariance of derivatives
    """
#(
    def __init__(
        self,
        diff: Optional[ndarray]=None,
        invariant: Optional[ndarray]=None,
        log_flag: bool=False,
    ) -> None:
        """
        """
        self._diff = diff
        self._invariant = invariant
        self._log_flag = log_flag


    def __add__(self, other):
        """
        Invariance of self(x)+other(x) or self(x)+other
        """
        if isinstance(other, DiffAtom):
            new_diff = self._diff | other._diff
            new_invariant = self._invariant & other._invariant
        else:
            new_diff = self._diff
            new_invariant = self._invariant
        return InvarianceAtom(new_diff, new_invariant)


    def __mul__(self, other):
        """
        Differentiate self(x)*other(x) or self(x)*other
        """
        if isinstance(other, DiffAtom):
            new_diff = self._diff | other._diff
            new_invariant = (
                np_all(self._diff, axis=0) & other._invariant
                |
                np_all(other._diff, axis=0) & self._invariant
            )
            new_log_flag = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_log_flag = self._log_flag
        return InvarianceAtom(new_diff, new_invariant, new_log_flag)


    def __rmul__(self, other):
        """
        Differentiat other*self(x)
        """
        return self


    def __truediv__(self, other):
        """
        Invariance self(x)/other(x) or self(x)/other
        """
        if isinstance(other, DiffAtom):
            new_diff = self._diff | other._diff
            new_invariant = np_all(other._diff, axis=0) & self._invariant
            new_log_flag = False
        else:
            new_diff = self._diff
            new_invariant = self._invariant
            new_log_flag = self._log_flag
        return InvarianceAtom(new_diff, new_invariant)


    def __rtruediv__(self, other):
        """
        Invariance of self(x)/other
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom(new_diff, new_invariant)


    __sub__ = __add__
    __radd__ = __add__
    __rsub = __add__


    def log(self):
        """
        Invariance of log(self(x))
        """
        new_diff = self._diff
        new_invariant = new_invariant if self._log_flag else (False & new_invariant)
        return InvarianceAtom(new_diff, new_invariant)


    def _unary(self):
        """
        Invariance of f(self(x))
        """
        new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom(new_diff, new_invariant)


    def _binary(self, other):
        """
        Invariance of f(self(x), other(x)) or f(self(x), other)
        """
        if isinstance(other, DiffAtom):
            new_diff = self._diff | other._diff
        elif:
            new_diff = self._diff
        new_invariant = False & self._invariant
        return InvarianceAtom(new_diff, new_invariant)


    exp = _unary
    __pow__ = _binary
#)





#(
def create_zero_atom(diff_shape: tuple[int, int]) -> DiffAtom:
    return DiffAtom(0, zeros(diff_shape))


def create_zero_invariant_atom(diff_shape: tuple[int, int]) -> DiffAtom:
    return DiffAtom(np.full(diff_shape, False, dtype=bool), np.full(diff_shape, False, dtype=bool))


def log(x): 
    if isinstance(x, Number)
        return math_log(x)
    else
        return x.log()


def exp(x): 
    if isinstance(x, Number):
        return math_exp(x)
    else:
        return x.exp()
#)





class Space:
    """
    """
#(
    def __init__(self) -> None:
        self._x: dict = {}
        self._func: Optional[Callable] = None


    @classmethod
    def from_equations(
        cls,
        equations: list[Equation],
        wrt_tokens_by_equations: dict[int, list[Token]],
        num_columns_to_eval: int,
        id_to_log_flag: dict[int, bool],
    ) -> Space:
        """
        """
        self = cls()

        all_tokens = set(collect_all_tokens(equations))
        min_shift = get_min_shift(all_tokens)
        max_shift = get_max_shift(all_tokens)
        self.shape_data = (
            1 + (get_max_quantity_id(all_tokens) or 0),
            -min_shift + num_columns_to_eval + max_shift,
        )
        self._t_zero = -min_shift
        self._columns_to_eval = (self._t_zero, self._t_zero + num_columns_to_eval - 1)

        self.wrt_tokens_by_equations = wrt_tokens_by_equations
        self._x = _create_atom_array(equations, self.wrt_tokens_by_equations, self._columns_to_eval, id_to_log_flag)

        self._xtrings = [ _create_audi_xtring(eqn, wrt_tokens_by_equations[eqn.id], num_columns_to_eval) for eqn in equations ]
        self._func_string = create_evaluator_func_string(self._xtrings)
        self._func = eval(self._func_string)

        return self


    @classmethod
    def _create_atom_array(
        cls,
        equations: list[Equation],
        wrt_tokens_by_equations: dict[int, list[Token]],
        columns_to_eval: tuple[int, int],
        id_to_log_flag: dict[int, bool],
    ) -> Space:
        """
        """
        x = {}
        for eqn in equations:
            wrt_tokens_here = wrt_tokens_by_equations[eqn.id]
            for tok in eqn.incidence:
                key = _create_audi_key(tok, eqn.id)
                diff = _diff_value_for_atom_from_incidence(tok, wrt_tokens_here)
                log_flag = id_to_log_flag.get(tok.quantity_id, False)
                x[key] = cls._create_atom(diff, log_flag, tok, columns_to_eval)
        return x


    def eval(self, data: ndarray) -> list[ndarray]:
        """
        """
        self._verify_data_array_shape(data.shape)
        DiffAtom._data = data
        return self._func(self._x, None)


    def _verify_data_array_shape(self, shape_data: ndarray) -> None:
        """
        """
        if shape_data!=self.shape_data:
            raise InvalidInputDataArrayShape(shape_data, self.shape_data)
#)





class DiffSpace(Space):
    """
    """
#(
    def _create_atom(diff, log_flag, tok, columns_to_eval):
        """
        """
        data_index = (
            slice(tok.quantity_id, tok.quantity_id+1),
            slice(columns_to_eval[0]+tok.shift, columns_to_eval[1]+tok.shift+1),
        )
        return DiffAtom(None, diff, data_index, log_flag)
#)





class InvarianceSpace(Space):
    """
    """
#(
    def _create_atom(diff, log_flag, tok, columns_to_eval):
        """
        """
        invariant = np.fill((diff.shape[0], 1), True)
        return InvarianceAtom(diff, invariant, log_flag)
#)




#(
def _diff_value_for_atom_from_incidence(
    token: Token,
    wrt_tokens: list[Token],
) -> Union[array, float]:
    """
    """
    if token in wrt_tokens:
        diff = zeros((len(wrt_tokens), 1))
        diff[wrt_tokens.index(token)] = 1
    else:
        diff = 0
    return diff


def _create_audi_xtring(equation: Equation, wrt_tokens: list[Token], num_columns_to_eval: int) -> str:
    """
    """
    xtring = equation.replace_equation_ref_in_xtring(equation.id)
    xtring = xtring.replace("[", "['").replace("]", "']")
    # Add np.zeros(___,___)+ to enforce the correct size of the resulting
    # diff in degenerate cases
    diff_shape = (len(wrt_tokens), num_columns_to_eval)
    xtring = f"create_zero_atom({diff_shape})+" + xtring
    xtring = "(" + xtring + ").diff"
    return xtring


def _create_audi_key(token: Token, equation_id: int) -> str:
    """
    """
    return X_REF_PATTERN.format( 
        quantity_id=token.quantity_id,
        shift=token.shift,
        equation_id=equation_id,
    )
#)



#
# Errors
#


class InvalidInputDataArrayShape(ListException):
    """
    """
    def __init__(self, shape_entered: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)


