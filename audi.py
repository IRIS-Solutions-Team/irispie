
from __future__ import annotations

from sys import exit
from numpy import log, exp, zeros, NaN, ndarray, array, vstack, full
from re import findall, compile
from typing import NamedTuple, Union
from collections.abc import Sequence
from numbers import Number

from .parser import DIFF_EQUATION_ID, parse_equation, extract_names
from .incidence import Incidence, Token, get_max_shift, get_min_shift, get_max_quantity_id
from .model import Equation, parse_equations


_VARIABLE_NAME_PATTERN = compile(r"\b[A-Za-z]\w*\b(?!\()")


InputValue = Union[Number, list[Number]]
InputValuesDict = dict[str, InputValue]


class MultilineException(Exception):
    #(
    def __init__(self, messages: list[str]) -> None:
        messages = "\n\n" + "\n".join([ "![modiphy] " + m for m in messages ]) + "\n"
        super().__init__(messages) 
    #)

class InvalidInputDataColumns(MultilineException):
    #(
    def __init__(self, invalid: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [
            f"Incorrect size of input data for '{n}': entered {l} values, needed {needed}" 
            for n, l in invalid.items()
        ]
        super().__init__(messages)
    #)


class InvalidInputDataArrayShape(MultilineException):
    #(
    def __init__(self, shape_entered: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [ f"Incorrect size of input data matrix: entered {shape_entered}, needed {needed}" ]
        super().__init__(messages)
    #)


class MissingInputValues(MultilineException):
    #(
    def __init__(self, missing: list[str]) -> None:
        messages = [ f"Missing input values for '{n}'" for n in missing ]
        super().__init__(messages)
    #)


class Unit():

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

        self._calculate_x()
        self._calculate_func(parsed_equations)


    @property
    def data_shape(self):
        return (self.num_data_rows, self.num_data_columns)


    def _preallocate_x(self) -> None:
        self.x = [ [None for _ in range(self.num_shifts)] for _ in range(self.num_data_rows) ]


    def _calculate_x(self) -> None:
        self._preallocate_x()
        for t in self._all_tokens:
            diff = [ _prepare_diff(t, w) for w in self._wrt_tokens ]
            data_index = (
                slice(t.quantity_id, t.quantity_id+1),
                slice(self.columns_to_eval[0]+t.shift, self.columns_to_eval[1]+t.shift+1),
            )
            log_flag = self._id_to_log_flag.get(t.quantity_id, False)
            self.x[t.quantity_id][self.t_zero+t.shift] = Unit(None, diff, data_index, log_flag)


    def _calculate_func(self, parsed_equations: list[str]) -> None:
        #(
        # Replace x[0][0][_] with x[0][0][i] 
        parsed_equations = ( e.replace(DIFF_EQUATION_ID, f"[{i}]") for i, e in enumerate(parsed_equations) )

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


def _verify_input_values_names(
        input_values: InputValuesDict,
        all_names: list[str], 
    ) -> None:
    """
    Verify that input_values dict contains all_names
    """
    if (missing := set(all_names).difference(input_values.keys())):
        raise MissingInputValues(missing)


def _data_matrix_from_input_values(
        input_values: InputValuesDict,
        name_to_id: dict[str, int],
        data_shape: tuple[int, int]
    ) -> ndarray:

    all_names = name_to_id.keys()
    _verify_input_values_names(input_values, all_names)
    _verify_input_values_len(input_values, all_names, data_shape[1])

    data = full(data_shape, NaN)
    for name, id in name_to_id.items():
        data[id, :] = input_values[name]

    return data


def diff_single(
        expression: str,
        wrt: list[str],
        *args,
        **kwargs,
    ) -> ndarray:
    """
    {== Differentiate an expression w.r.t. to selected variables ==}

    ## Syntax

        d = diff_single(expression, wrt, input_values, num_columns_to_eval=1, log_list=None)

    ## Input arguments

    * `expression`: `str`
    >
    > Expression that will be differentiated with respect to the list of
    > variables given by `wrt`
    >

    * `wrt`: `list[str]`
    >
    > List of lists of variables with respect to which the corresponding
    > expression will be differentiated
    >

    * `input_values`: `dict[str, Number]`
    >
    > Dictionary of values at which the expressions will be differentiated
    >

    ## Output arguments

    * `d`: `ndarray`
    >
    > List of arrays with the numerical derivatives; d[i][j,:] is an array
    > of derivatives of the ith-expression w.r.t. to j-th variable
    >
    """
    diff, *args = diff_multiple([expression], [wrt], *args, **kwargs)
    return diff[0], *args


def diff_multiple(
        expressions: list[str],
        wrts: list[list[str]],
        input_values: InputValuesDict,
        num_columns_to_eval: int=1,
        log_list: Optional[list[str]]=None,
    ) -> list[ndarray]:
    """
    {== Differentiate a list of expressions w.r.t. to selected variables all at once ==}

    ## Syntax

        d = diff_multiple(expressions, wrts, input_values, num_columns_to_eval=1, log_list=None)

    ## Input arguments

    * `expressions`: `list[str]`
    >
    > List of expressions; expressions[i] will be differentiated with
    > respect to the list of variables given by `wrt[i]`
    >

    * `wrts`: `list[list[str]]`
    >
    > List of lists of variables with respect to which the corresponding
    > expression will be differentiated
    >

    * `input_values`: `dict[str, Number]`
    >
    > Dictionary of values at which the expressions will be differentiated
    >

    ## Output arguments

    * `d`: `list[ndarray]`
    >
    > List of arrays with the numerical derivatives; d[i][j,:] is an array
    > of derivatives of the ith-expression w.r.t. to j-th variable
    >
    """

    all_names = extract_names(" ".join(expressions))

    name_to_id = { name:i for i, name in enumerate(all_names) }

    id_to_log_flag = (
        { name_to_id[n]:True for n in log_list if n in name_to_id } 
        if log_list is not None else {}
    )

    expressions = [ Equation(i, e) for i, e in enumerate(expressions) ]
    parsed_expressions, all_incidences = parse_equations(expressions, name_to_id)

    # Extract tokens (quantity_id, shift) from incidences (equation_id, (quantity_id, shift))
    all_tokens = set(i.token for i in all_incidences)

    # Translate name-shifts to tokens, preserving the order
    # Use output argument #3 from parse_equation which is a list with
    # the tokens in the same order as the name-shifts 
    wrt_tokens: list[list[Token]] = [ 
        parse_equation(" ".join(w), name_to_id)[3] for w in wrts
    ]

    print(wrt_tokens[0])

    space = Space( 
        parsed_expressions,
        all_tokens,
        wrt_tokens,
        id_to_log_flag,
        num_columns_to_eval,
    )

    data = _data_matrix_from_input_values(input_values, name_to_id, space.data_shape)

    return space.eval(data), space, name_to_id


# if __name__=="__main__":
    # expressions = ["log(a) + c{+1}*b{+1}^2 + b", "c{-1}*d + 2*d{+1}"]
    # wrts = [["a", "b{+1}", "c{-1}"], ["c{-1}", "d", "d{+1}"]]
    # input_values = {"a":5, "b":2, "c":3, "d":-4,}
# 
    # diff, space, name_to_id = diff_multiple(expressions, wrts, input_values, 3, ["a"])

