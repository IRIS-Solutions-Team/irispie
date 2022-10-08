
#(
from __future__ import annotations


from numpy import log as np_log, exp as np_exp, zeros, NaN, ndarray
from re import findall, compile
from typing import NamedTuple, Optional, Union, Callable
from numbers import Number

from .exceptions import ListException

from .eqman import (
    EQUATION_ID_PLACEHOLDER, XtringsT,
    create_evaluator_func_string
)

from .incidence import (
    Incidence, Token, get_max_shift, 
    get_min_shift, get_max_quantity_id
)
#)



class InvalidInputDataArrayShape(ListException):
#{
    def __init__(self, shape_entered: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)
#}


class Atom():
#{
    _data: ndarray  = None

    def __init__(
            self, 
            value: Optional[Number]=None,
            diff: Optional[Union[Number, ndarray]]=None,
            data_index: Optional[tuple[slice, slice]]=None,
            log_flag: bool=False,
            human: str='',
        ) -> None:

        self._value = value
        self._diff = diff
        self._data_index = data_index
        self._log_flag = log_flag
        self.human = human


    def __getitem__(self, index):
        return Atom(
            self._value,
            self._diff[index],
            self._data_index,
            self._log_flag,
        )


    @property
    def value(self):
        return self._value if self._value is not None else self._data[self._data_index]


    @property
    def diff(self):
        return self._diff if not self._log_flag else self._diff * self.value


    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return Atom(new_value, new_diff)


    def __add__(self, other):
        if isinstance(other, Atom):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return Atom(new_value, new_diff)


    def __sub__(self, other):
        if isinstance(other, Atom):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        return Atom(new_value, new_diff)


    def __mul__(self, other):
        self_value = self.value
        self_diff = self.diff
        if isinstance(other, Atom):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value * other_value
            new_diff = self_diff * other_value + self_value * other_diff
        else:
            new_value = self_value * other
            new_diff = self_diff * other
        return Atom(new_value, new_diff)


    def __truediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        if isinstance(other, Atom):
            other_value = other.value
            other_diff = other.diff
            new_value = self_value / other_value
            new_diff = (self_diff*other_value - self_value*other_diff) / (other_diff**2)
        else:
            new_value = self_value / other
            new_diff = self_diff / other
        return Atom(new_value, new_diff)


    def __rtruediv__(self, other):
        self_value = self.value
        self_diff = self.diff
        new_value = other / self_value
        new_diff = -other*self_diff / (self_value**2)
        return Atom(new_value, new_diff)


    def __pow__(self, other):
        """
        Differentiate self**other 
        """
        if isinstance(other, Atom):
            # self(x)**other(x)
            # d[self(x)**other(x)] = d[self(x)**other] + d[self**other(x)]
            new_value = self.value**other.value
            _, new_diff_exp = self._power(other.value) # self(x)**other
            _, new_diff_pow = other._exponential(self.value) # self**other(x)
            new_diff = new_diff_exp + new_diff_pow
        else:
            # self(x)**other
            new_value, new_diff = self._power(other)
        return Atom(new_value, new_diff)


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
        return Atom(new_value, new_diff)


    def exp(self):
        new_value = np_exp(self.value)
        new_diff = new_value * self.diff
        return Atom(new_value, new_diff)
#}


def log(x): 
#(
    if isinstance(x, Atom):
        return x.log()
    else:
        return np_log(x)
#)


def exp(x): 
#(
    if isinstance(x, Atom):
        return x.exp()
    else:
        return np_exp(x)
#)


def diff(x): 
#(
    if isinstance(x, Atom):
        return x.diff
    else:
        return 0
#)


def _prepare_diff_value_for_atom( 
    token: Token, 
    wrt_tokens: list[Token],
) -> Union[float, ndarray]:
#(
    if token in wrt_tokens:
        num_wrt = len(wrt_tokens)
        diff = zeros((num_wrt, 1), dtype=int)
        index = wrt_tokens.index(token)
        diff[index, :] = 1
        return diff
    else:
        return 0
#)

def _get_human_for_atom(
    t: Token,
    id_to_name: dict[int, str],
) -> str:
#(
    human = id_to_name.get(t.quantity_id, f'x{t.quantity_id:g}')
    if t.shift:
        human = human + f'{{{t.shift:+g}}}'
    return human
#)


def _create_atom_array(
    all_tokens: set[Token],
    wrt_tokens: list[list[Token]],
    id_to_log_flag: dict[int, bool]={},
    t_zero: int=0,
    columns_to_eval: tuple(int, int)=(0, 0),
    id_to_name: Optional[dict[int, name]]=None,
) -> list[list[Atom]]:
#(

    num_equations = len(wrt_tokens)
    max_shift = get_max_shift(all_tokens)
    min_shift = get_min_shift(all_tokens)
    num_shifts = (max_shift - min_shift + 1) if (max_shift is not None) and (min_shift is not None) else 0
    num_data_rows = _get_num_data_rows(all_tokens)

    # Preallocate x as a list of list of Nones
    x: list[list[Atom]] = [
        [None for _ in range(num_shifts)] 
        for _ in range(num_data_rows)
    ]

    # Populate x with Atom objects
    for t in all_tokens:
        diff = [ _prepare_diff_value_for_atom(t, w) for w in wrt_tokens ]
        data_index = (
            slice(t.quantity_id, t.quantity_id+1),
            slice(columns_to_eval[0]+t.shift, columns_to_eval[1]+t.shift+1),
        )
        log_flag = id_to_log_flag.get(t.quantity_id, False)
        human = _get_human_for_atom(t, id_to_name)
        x[t.quantity_id][t_zero+t.shift] = Atom(None, diff, data_index, log_flag, human)

    return x
#)


def _get_num_data_rows(tokens: set[Token]) -> int:
    return 1 + (get_max_quantity_id(tokens) or 0)


class Space:
    #{

    def __init__(
        self, 
        equations: XtringsT,
        all_tokens: set[Token], 
        wrt_tokens: list[list[Token]], 
        id_to_log_flag: dict[int, bool]={},
        num_columns_to_eval: int=1,
        id_to_name: Optional[dict[int, str]]=None
    ) -> None:

        self.num_data_rows = _get_num_data_rows(all_tokens)
        min_shift = get_min_shift(all_tokens)
        max_shift = get_max_shift(all_tokens)
        self.num_data_columns = -min_shift + num_columns_to_eval + max_shift
        self._t_zero = -min_shift
        self._columns_to_eval = (self._t_zero, self._t_zero + num_columns_to_eval - 1)

        self._x = _create_atom_array( 
            all_tokens,
            wrt_tokens,
            id_to_log_flag,
            self._t_zero,
            self._columns_to_eval,
            id_to_name,
        )

        diff_sizes = [ (len(w), num_columns_to_eval) for w in wrt_tokens ]

        self._func = _create_evaluator(equations, diff_sizes)


    def eval(self, data: ndarray) -> list[ndarray]:
        self._verify_data_array_shape(data)
        Atom._data = data
        return self._func(self._x, self._t_zero)


    @property
    def data_shape(self) -> tuple[int, int]:
        return self.num_data_rows, self.num_data_columns


    def _verify_data_array_shape(self, data: ndarray) -> None:
        if data.shape!=self.data_shape:
            raise InvalidInputDataArrayShape(data.shape, self.data_shape)
    #}


def _create_equation_for_evaluator(
    equation_id: int,
    equation: str,
    diff_sizes: tuple[int, int],
) -> str:
#(
    # Replace x[0][0][_] with x[0][0][i] in parsed equations
    equation = equation.replace(EQUATION_ID_PLACEHOLDER, f"[{equation_id}]") 
    # Wrap each equation in diff(___) 
    equation = "diff(" + equation + ")"
    # Add ndarray(___,___)+ to enforce the correct size of the resulting
    # diff in degenerate cases
    equation = f"zeros({diff_sizes},dtype=int)+" + equation
    return equation.replace(" ","")
#)


def _create_evaluator(
    equations: XtringsT,
    diff_sizes: list[tuple[int, int]],
) -> Callable:
#(

    equations = (
        _create_equation_for_evaluator(i[0], *i[1]) 
        for i in enumerate(zip(equations, diff_sizes))
    )
    func_string = create_evaluator_func_string(equations)
    func = eval(func_string)
    func.string = func_string
    return func
#)

