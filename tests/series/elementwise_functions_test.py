
import irispie as ir
import irispie.series.functions as isf
import numpy as _np
import scipy as _sp
from irispie.series._elementwise import _ONE_ARG_FUNCTION_DISPATCH, _TWO_ARGS_FUNCTION_DISPATCH

import random as rn
import pytest


span = ir.qq(2021,1) >> ir.qq(2025,4)

x = ir.Series(periods=span, num_variants=2, func=lambda: rn.uniform(0.1, 0.9))


parameter_names = ("func_name", )
parameter_values = [ (n, ) for n in _ONE_ARG_FUNCTION_DISPATCH.keys() ]
@pytest.mark.parametrize(parameter_names, parameter_values, )
def test_elementwise_one_arg(func_name, ):
    y1 = x.copy()
    getattr(y1, func_name)()
    #
    y2 = getattr(isf, func_name)(x, )
    #
    y3 = getattr(ir, func_name)(x, )
    #
    implementation_name = _ONE_ARG_FUNCTION_DISPATCH[func_name]
    expected_data = eval(f"{implementation_name}(x.data, )", )
    #
    assert _np.all(y1.data == expected_data)
    assert _np.all(y2.data == expected_data)
    assert _np.all(y3.data == expected_data)


parameter_names = ("func_name", )
parameter_values = [ (n, ) for n in _TWO_ARGS_FUNCTION_DISPATCH.keys() ]
@pytest.mark.parametrize(parameter_names, parameter_values, )
def test_elementwise_two_args(func_name, ):
    y1 = x.copy()
    getattr(y1, func_name)(1, )
    #
    y2 = getattr(isf, func_name)(x, 1, )
    #
    y3 = getattr(ir, func_name)(x, 1, )
    #
    implementation_name = _TWO_ARGS_FUNCTION_DISPATCH[func_name]
    expected_data = eval(f"{implementation_name}(x.data, 1, )", )
    #
    assert _np.all(y1.data == expected_data)
    assert _np.all(y2.data == expected_data)
    assert _np.all(y3.data == expected_data)


if __name__ == "__main__":
    test_elementwise_one_arg("log", )
    test_elementwise_two_args("minimum", )


