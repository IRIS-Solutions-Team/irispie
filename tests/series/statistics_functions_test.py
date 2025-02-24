
import irispie as ir
import irispie.series.functions as isf
from irispie.series._statistics import __all__ as __all__statistics
import numpy as np
import random as rn
import pytest

rn.seed(0)

span = ir.qq(2021,1) >> ir.qq(2025,4)

x1 = ir.Series(periods=span, func=rn.gauss, )
x10 = ir.Series(periods=span, func=rn.gauss, num_variants=10, )

extra_args_functions = ("percentile", "quantile", "nanpercentile", "nanquantile", )

parameter_names = ("func_name", )
parameter_values = [ (n, ) for n in __all__statistics if n not in extra_args_functions ]

@pytest.mark.parametrize(parameter_names, parameter_values, )
def test_statistics(func_name, ):
    # Run the method on the copy of the Series object
    y21 = x10.copy()
    getattr(y21, func_name, )()
    # Run the function creating a new Series object
    y22 = getattr(isf, func_name, )(x10, )
    # Run the NP function on the Series data
    expected_data = getattr(np, func_name, )(x10.data, axis=1, ).reshape(-1, 1, )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


def test_percentile_axis_1():
    q = (0, 50, 100, )
    # Run the method on the copy of the Series object
    y21 = x10.copy()
    y21.percentile(q, )
    # Run the function creating a new Series object
    y22 = isf.percentile(x10, q, )
    # Run the NP function on the Series data
    expected_data = np.percentile(x10.data, q, axis=1, ).T.reshape(-1, len(q), )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


def test_percentile_axis_0():
    q = (0, 50, 100, )
    # Run the function creating a new Series object
    y22 = isf.percentile(x10, q, axis=0, )
    # Run the NP function on the Series data
    expected_data = np.percentile(x10.data, q, axis=0, ).T.tolist()
    #
    assert all(y == e for y, e in zip(y22, expected_data))


def test_quantiles():
    q = (0, 0.5, 1, )
    # Run the method on the copy of the Series object
    y21 = x10.copy()
    y21.quantile(q, )
    # Run the function creating a new Series object
    y22 = isf.quantile(x10, q, )
    # Run the NP function on the Series data
    expected_data = np.quantile(x10.data, q, axis=1, ).T.reshape(-1, len(q), )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


def test_quantile_axis_0():
    q = (0, 0.5, 1, )
    # Run the function creating a new Series object
    y22 = isf.quantile(x10, q, axis=0, )
    # Run the NP function on the Series data
    expected_data = np.quantile(x10.data, q, axis=0, ).T.tolist()
    #
    assert all(y == e for y, e in zip(y22, expected_data))


