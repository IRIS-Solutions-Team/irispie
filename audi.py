
#(
from __future__ import annotations



from numpy import (
    log, exp, zeros, NaN, 
    ndarray, array, vstack, full
)


from re import findall, compile

from typing import NamedTuple, Union

from collections.abc import Sequence

from numbers import Number



from .exceptions import ListException

from .parser import DIFF_EQUATION_ID

from .incidence import (
    Incidence, Token, get_max_shift, 
    get_min_shift, get_max_quantity_id
)
#)


InputValue = Union[Number, list[Number]]

InputValuesDict = dict[str, InputValue]


class InvalidInputDataColumns(ListException):
    #(
    def __init__(self, invalid: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [
            f"Incorrect size of input data for '{n}': entered {l} values, needed {needed}" 
            for n, l in invalid.items()
        ]
        super().__init__(messages)
    #)


class InvalidInputDataArrayShape(ListException):
    #(
    def __init__(self, shape_entered: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)
    #)


class Unit():

    _data: ndarray  = None

    def __init__(
            self, 
            value: Optional[Number]=None,
            diff: Optional[Union[Number, ndarray]]=None,
            data_index: Optional[tuple[slice, slice]]=None,
            log_flag: bool=False,
        ) -> None:

        self._value = value
        self.diff = diff
        self._data_index = data_index
        self._log_flag = log_flag


    def __getitem__(self, diff_index):
        diff = self.diff[diff_index]
        return Unit(
            self._value,
            self.diff[diff_index] if not self._log_flag else self.diff[diff_index]*self.value,
            self._data_index
        )

    @property
    def value(self):
        return self._value if self._value is not None else self._data[self._data_index]

    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return Unit(new_value, new_diff)


    def __add__(self, other):
        if isinstance(other, Unit):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return Unit(new_value, new_diff)


    def __sub__(self, other):
        if isinstance(other, Unit):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        return Unit(new_value, new_diff)


    def __mul__(self, other):
        if isinstance(other, Unit):
            new_value = self.value * other.value
            new_diff = self.diff * other.value + self.value * other.diff
        else:
            new_value = self.value * other
            new_diff = self.diff * other
        return Unit(new_value, new_diff)


    def __pow__(self, other):
        """
        Differentiate self ** other 
        """
        if isinstance(other, Unit):
            # f(x) ** g(x)
            new_value = self.value ** other.value
            _, new_diff_powf = self._exp(other.value)
            _, new_diff_expf = self._pow(other.value)
            new_diff = new_diff_powf + new_diff_expf
        else:
            # f(x) ** k
            new_value, new_diff = self._pow(other)
        return Unit(new_value, new_diff)

    __xor__ = __pow__


    def _exp(self, other):
        """
        Differentiate exponential function k ** f(x)
        """
        new_value = other ** self.value
        new_diff = other ** self.value * log(other) * self.diff
        return new_value, new_diff


    def _pow(self, other):
        """
        Differentiate power function f(x) ** k
        """
        new_value = self.value ** other
        new_diff = other * (self.value ** (other-1)) * self.diff
        return new_value, new_diff


    __rpow__ = _exp

    __rmul__ = __mul__

    __radd__ = __add__


    def __rsub__(self, other):
        return self.__neg__().__sub__(-other)


    def log(self):
        new_value = log(self.value)
        new_diff = 1 / self.value * self.diff
        return Unit(new_value, new_diff)


    def exp(self):
        new_value = exp(self.value)
        new_diff = new_value * self.diff
        return Unit(new_value, new_diff)


class Space:

    def __init__(
            self, 
            parsed_equations: list[str], 
            all_tokens: set[Token], 
            wrt_tokens: list[list[Token]], 
            id_to_log_flag: dict[int,bool]={},
            num_columns_to_eval: int=1,
        ) -> None:

        self._all_tokens = all_tokens
        self._wrt_tokens = wrt_tokens
        self._unit = Unit
        self._id_to_log_flag = id_to_log_flag

        self.num_data_rows = 1 + get_max_quantity_id(self._all_tokens)
        max_shift = get_max_shift(self._all_tokens)
        min_shift = get_min_shift(self._all_tokens)
        self.num_shifts = max_shift - min_shift + 1
        self.num_data_columns = -min_shift + num_columns_to_eval + max_shift
        self.t_zero = -min_shift
        self.columns_to_eval = (self.t_zero, self.t_zero + num_columns_to_eval - 1)

        self._populate_x()
        self._populate_func(parsed_equations)


    @property
    def data_shape(self):
        return (self.num_data_rows, self.num_data_columns)


    def _preallocate_x(self) -> None:
        self.x: list[list] = [ 
            [None for _ in range(self.num_shifts)] 
            for _ in range(self.num_data_rows)
        ]


    def _populate_x(self) -> None:
        self._preallocate_x()
        for t in self._all_tokens:
            diff = [ self._prepare_diff(t, w) for w in self._wrt_tokens ]
            data_index = (
                slice(t.quantity_id, t.quantity_id+1),
                slice(self.columns_to_eval[0]+t.shift, self.columns_to_eval[1]+t.shift+1),
            )
            log_flag = self._id_to_log_flag.get(t.quantity_id, False)
            self.x[t.quantity_id][self.t_zero+t.shift] = Unit(None, diff, data_index, log_flag)


    def _populate_func(self, parsed_equations: list[str]) -> None:
        #(
        # Replace x[0][0][_] with x[0][0][i] 
        parsed_equations = ( 
            e.replace(DIFF_EQUATION_ID, f"[{i}]") 
            for i, e in enumerate(parsed_equations)
        )

        # Wrap each equation in (___).diff
        wrapped_equations = ( f"({e}).diff," for e in parsed_equations )

        # Create a lambda function string putting all equations in a list
        func_string = "lambda x, t: [" + " ".join(wrapped_equations) + "]"

        # Compile a lambda
        self.func = eval(func_string)
        #)


    def eval(self, data: ndarray) -> list[ndarray]:
        self._verify_data_array_shape(data)
        self._unit._data = data
        return self.func(self.x, self.t_zero)


    def _verify_data_array_shape(self, data: ndarray) -> None:
        if data.shape!=self.data_shape:
            raise InvalidInputDataArrayShape(data.shape, self.data_shape)



    @staticmethod
    def _prepare_diff(token: Token, wrt_tokens: list[Token]) -> Union[float, ndarray]:
            if token in wrt_tokens:
                num_wrt = len(wrt_tokens)
                diff = zeros((num_wrt, 1), dtype=int)
                diff[wrt_tokens.index(token), :] = 1
                return diff
            else:
                return 0


def _verify_input_values_len(
        input_values: InputValuesDict,
        all_names: list[str], 
        num_data_columns: int, 
    ) -> None:
    """
    Verify that all input values are scalars or lists of length num_data_columns
    """
    #(
    def spec_len(x: Optional[InputValue]) -> Optional[int]:
        match x:
            case None:
                return None
            case [*__]:
                return len(x)
            case _:
                return 0
    invalid = { 
        n: l for n in all_names 
        if (l := spec_len(input_values.get(n))) not in {0, num_data_columns}
    }
    if invalid:
        raise InvalidInputDataColumns(invalid, num_data_columns)
    #)

