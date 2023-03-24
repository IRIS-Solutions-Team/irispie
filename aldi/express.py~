
from __future__ import annotations
from typing import Optional
from numpy import ndarray, full, NaN

from .exceptions import ListException
from .aldi import Space
from .types import InputValueT, InputValuesT
from .equations import Equation, xtrings_from_equations, xtring_from_human, names_from_human
from .incidence import Token


class MissingInputValues(ListException):
#(
    def __init__(self, missing: list[str]) -> None:
        messages = [ f"Missing input values for '{n}'" for n in missing ]
        super().__init__(messages)
#)


class InvalidInputDataColumns(ListException):
#{
    def __init__(self, invalid: tuple[int, int], needed: tuple[int, int]) -> None:
        messages = [
            f"Incorrect size of input data for '{n}': entered {l} values, needed {needed}" 
            for n, l in invalid.items()
        ]
        super().__init__(messages)
#}


def _verify_input_values_names(
    input_values: InputValuesT,
    all_names: list[str], 
) -> None:
#(
    """
    Verify that input_values dict contains all_names
    """
    if (missing := set(all_names).difference(input_values.keys())):
        raise MissingInputValues(missing)
#)


def _verify_input_values_len(
    input_values: InputValuesT,
    all_names: list[str], 
    num_data_columns: int, 
) -> None:
#(
    """
    Verify that all input values are scalars or lists of length num_data_columns
    """
    def spec_len(x: Optional[InputValueT]) -> Optional[int]:
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


def data_matrix_from_input_values(
    input_values: InputValuesT,
    name_to_id: dict[str, int],
    data_shape: tuple[int, int],
) -> ndarray:
#(
    all_names = name_to_id.keys()
    _verify_input_values_names(input_values, all_names)
    _verify_input_values_len(input_values, all_names, data_shape[1])

    data = full(data_shape, NaN)
    for name, id in name_to_id.items():
        data[id, :] = input_values[name]

    return data
#)


def diff_single(
    expression: str,
    wrt: list[str],
    *args,
    **kwargs,
) -> ndarray:
#(
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
#)


def diff_multiple(
    expressions: list[str],
    wrts: list[list[str]],
    input_values: InputValuesT,
    num_columns_to_eval: int=1,
    log_list: Optional[list[str]]=None,
) -> list[ndarray]:
#(
    """
    {== Differentiate a list of expressions w.r.t. to selected variables all at once ==}

    ## Syntax

        d = diff_multiple(expressions, wrts, input_values, num_columns_to_eval=1, log_list=None)

    ## Input arguments

    __`expressions`__: `list[str]`
    >
    > List of expressions; expressions[i] will be differentiated with
    > respect to the list of variables given by `wrt[i]`
    >

    __`wrts`__: `list[list[str]]`
    >
    > List of lists of variables with respect to which the corresponding
    > expression will be differentiated
    >

    __`input_values`__: `dict[str, Number]`
    >
    > Dictionary of values at which the expressions will be differentiated
    >

    ## Output arguments

    __`d`__: `list[ndarray]`
    >
    > List of arrays with the numerical derivatives; d[i][j,:] is an array
    > of derivatives of the ith-expression w.r.t. to j-th variable
    >
    """

    if log_list is None:
        log_list = []

    # Retrieve all names from equations and all names from wrts
    all_names = names_from_human(" ".join(expressions) + " " + " ".join(w for e in wrts for w in e))

    # Create map {name: qid}
    name_to_id = { name: i for i, name in enumerate(all_names) }

    # Create map {qid: True, ...} for all names in name_to_id
    id_to_logly = (
        { name_to_id[name]: (name in log_list) for name in name_to_id }
    )

    expressions = [ Equation(i, e) for i, e in enumerate(expressions) ]
    xtrings, incidences = xtrings_from_equations(expressions, name_to_id)

    # Extract all tokens (qid, shift) from incidences (eid, (qid, shift))
    all_tokens = set(i.token for i in incidences)

    # Translate name-shifts to tokens, preserving the order
    # Use output argument #3 from xtring_from_human which is a list with
    # the tokens in the same order as the name-shifts 
    wrt_tokens: list[list[Token]] = [ 
        xtring_from_human(" ".join(w), name_to_id)[3] for w in wrts
    ]

    # Make sure all_tokens include all wrt_tokens
    all_tokens.update(w for e in wrt_tokens for w in e)

    id_to_name = { id: name for name, id in name_to_id.items() }

    space = Space( 
        xtrings,
        all_tokens,
        wrt_tokens,
        id_to_logly,
        num_columns_to_eval,
        id_to_name
    )

    data = data_matrix_from_input_values(input_values, name_to_id, space.data_shape)

    return space.eval(data), space, data, name_to_id
    # return space, data
#)

