"""
Algorithmic differentiator
"""


#[
from __future__ import annotations

from typing import (Self, Callable, Protocol, )
from numbers import (Number, )
from collections.abc import (Iterable, Sequence, )
import numpy as np_

from ..exceptions import ListException

from ..aldi import (finite_differentiators as af_, adaptations as aa_, )
from .. import (equations as eq_, incidence as in_, )
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
    _data_context: np_.ndarray | None = None
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
        diff: np_.ndarray | Number,
        data_index: tuple[int, slice],
        logly: bool,
        /,
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
    def zero_atom(
        cls,
        diff_shape: tuple[int, int],
    ) -> Self:
        return cls.no_context(0, np_.zeros(diff_shape), False)

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
            other_value**self.value * np_.log(other_value) * self.diff 
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
        new_value = np_.log(self.value)
        new_diff = 1 / self.value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def _exp_(self, /, ) -> Self:
        """
        Differentiate exp(self)
        """
        new_value = np_.exp(self.value)
        new_diff = new_value * self.diff
        return type(self).no_context(new_value, new_diff, False)

    def _sqrt_(self, /, ) -> Self:
        """
        Differentiate sqrt(self)
        """
        new_value = np_.sqrt(self.value)
        new_diff = 0.5 / np_.sqrt(self.diff)
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
            orig_value = np_.array(orig_value, dtype=float)
            orig_diff = np_.array(orig_diff, dtype=float)
        new_value = np_.copy(orig_value)
        inx_floor = orig_value == floor
        inx_below = orig_value < floor
        inx_above = orig_value > floor
        new_value[inx_below] = floor
        new_diff = np_.copy(orig_diff)
        multiplier = np_.copy(orig_value)
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
        equations: eq_.Equations,
        atom_factory: AtomFactoryProtocol,
        /,
        eid_to_wrts: dict[int, tuple[Any]],
        qid_to_logly: dict[int, bool] | None,
        num_columns_to_eval: int,
        custom_functions: dict | None,
    ) -> Self:
        """
        """
        equations = list(equations)
        all_tokens = set(eq_.generate_all_tokens_from_equations(equations))
        self.min_shift = in_.get_min_shift(all_tokens)
        self.max_shift = in_.get_max_shift(all_tokens)
        self.shape_data = (
            1 + (in_.get_max_qid(all_tokens) or 0),
            -self.min_shift + num_columns_to_eval + self.max_shift,
        )
        #
        t_zero = -self.min_shift
        self._columns_to_eval = (t_zero, t_zero + num_columns_to_eval - 1)
        #
        self._populate_atom_dict(
            atom_factory,
            equations,
            eid_to_wrts,
            qid_to_logly,
            self._columns_to_eval,
        )
        #
        custom_functions = { 
            k: af_.finite_differentiator(v) 
            for k, v in custom_functions.items()
        } if custom_functions else None
        custom_functions = aa_.add_function_adaptations_to_custom_functions(custom_functions)
        custom_functions["Atom"] = Atom
        #
        get_diff_shape_for_eid = lambda eid: (
            len(eid_to_wrts[eid]), num_columns_to_eval,
        )
        xtrings = [
            _create_aldi_xtring(eqn, get_diff_shape_for_eid(eqn.id), )
            for eqn in equations
        ]
        self._func_string = eq_.create_equator_func_string(xtrings)
        self._func = eval(self._func_string, custom_functions, )

    def _populate_atom_dict(
        self,
        atom_factory: AtomFactoryProtocol,
        equations: eq_.Equations,
        eid_to_wrts: dict[int, Iterable[Any]],
        qid_to_logly: dict[int, bool] | None,
        columns_to_eval: tuple[int, int],
        /,
    ) -> None:
        """
        * atom_factory -- Implement create_diff_for_token, create_data_index_for_token, get_logly_for_token
        * equations -- List of equations
        * eid_to_wrts -- Dictionary of eid to wrts
        * qid_to_logly -- Dictionary of qid to logly
        * columns_to_eval -- Tuple of (first, last) column indices to be evaluated
        """
        x = {}
        for eqn in equations:
            for tok in eqn.incidence:
                #
                # Create string representing the qid, shift and eid
                # as x["{qid},t+{shift},{eid}"]
                key = _create_aldi_key(tok, eqn.id)
                #
                # Call the atom factory to create the atom attributes
                diff = atom_factory.create_diff_for_token(tok, eid_to_wrts[eqn.id], )
                data_index = atom_factory.create_data_index_for_token(tok, columns_to_eval, )
                logly = qid_to_logly.get(tok.qid, False) if qid_to_logly else False
                #
                # Store the atom in the x dictionary
                x[key] = Atom.in_context(diff, data_index, logly, )
        self._x = x

    def eval(
        self,
        data_array: np_.ndarray,
        steady_array: np_.ndarray,
    ) -> Iterable[Atom]:
        """
        Evaluate and return a list of final atoms, one for each equation
        """
        self._verify_data_array_shape(data_array.shape)
        Atom._data_context = data_array
        output = self._func(self._x, None, steady_array, )
        Atom._data_context = None
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

    def eval_diff_to_array(
        self,
        *args,
    ) -> np_.array:
        """
        Evaluate and return array of diffs
        """
        return np_.vstack([
            x.diff for x in self.eval(*args)
            if hasattr(x, "diff")
        ])

    def _verify_data_array_shape(self, shape_data: np_.ndarray) -> None:
        """
        """
        if shape_data[0]>=self.shape_data[0] and shape_data[1]>=self.shape_data[1]:
            return
        raise InvalidInputDataArrayShape(shape_data, self.shape_data)
    #]


def _create_aldi_xtring(
    equation: eq_.Equation,
    diff_shape: tuple[int, int],
    /,
) -> str:
    """
    """
    #[
    xtring = equation.replace_equation_ref_in_xtring(equation.id)
    xtring = xtring.replace("[", "['").replace("]", "']")
    sign = "+" if not xtring.startswith("-") and not xtring.startswith("+") else ""
    return f"Atom.zero_atom({diff_shape})" + sign + xtring
    #]


def _create_aldi_key(
    token: in_.Token,
    eid: int,
) -> str:
    """
    Craete a hashable representation of a token in an Audi expression
    """
    #[
    return eq_.X_REF_PATTERN.format( 
        qid=token.qid,
        shift=token.shift,
        eid=eid,
    )
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
        token: in_.Token,
        wrts: tuple[Any, ...],
    ) -> np_.ndarray:
        """
        Create a diff for an atom representing a given token
        """
        ...

    def create_data_index_for_token(
        self,
        token: in_.Token,
        columns_to_eval: tuple[int, int],
    ) -> tuple[int, slice]:
        """
        Create a data index for an atom representing a given token
        """
        ...
    #]


