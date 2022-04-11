
from __future__ import annotations

from .exceptions import ListException
from .audi import Space
from .parser import parse_equation, extract_names
from .model import Equation, parse_equations
from .incidence import Token



class MissingInputValues(ListException):
    #(
    def __init__(self, missing: list[str]) -> None:
        messages = [ f"Missing input values for '{n}'" for n in missing ]
        super().__init__(messages)
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

    name_to_id = { name: i for i, name in enumerate(all_names) }

    # Create map {quantity_id: True, ...}
    id_to_log_flag = (
        { name_to_id[name]: True for name in log_list if name in name_to_id } 
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

    space = Space( 
        parsed_expressions,
        all_tokens,
        wrt_tokens,
        id_to_log_flag,
        num_columns_to_eval,
    )

    data = _data_matrix_from_input_values(input_values, name_to_id, space.data_shape)

    return space.eval(data), space, name_to_id

