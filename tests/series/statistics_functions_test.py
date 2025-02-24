
import irispie as ir
import irispie.series.functions as isf
from irispie.series._statistics import __all__ as __all__statistics
import numpy as np
import random as rn
import pytest

rn.seed(0)

span = ir.qq(2021,1) >> ir.qq(2025,4)

x1 = ir.Series(periods=span, func=rn.gauss, )
x2 = ir.Series(periods=span, func=rn.gauss, num_variants=10, )

extra_args_functions = ("percentile", "quantile", "nanpercentile", "nanquantile", )

parameter_names = ("func_name", )
parameter_values = [ (n, ) for n in __all__statistics if n not in extra_args_functions ]

@pytest.mark.parametrize(parameter_names, parameter_values, )
def test_statistics(func_name, ):
    # Run the method on the copy of the Series object
    y21 = x2.copy()
    getattr(y21, func_name, )()
    # Run the function creating a new Series object
    y22 = getattr(isf, func_name, )(x2, )
    # Run the NP function on the Series data
    expected_data = getattr(np, func_name, )(x2.data, axis=1, ).reshape(-1, 1, )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


def test_percentile():
    q = (10, 50, 90, )
    # Run the method on the copy of the Series object
    y21 = x2.copy()
    y21.percentile(q, )
    # Run the function creating a new Series object
    y22 = isf.percentile(x2, q, )
    # Run the NP function on the Series data
    expected_data = np.percentile(x2.data, q, axis=1, ).reshape(-1, len(q), )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


def test_quantiles():
    q = (0.1, 0.5, 0.9, )
    # Run the method on the copy of the Series object
    y21 = x2.copy()
    y21.quantile(q, )
    # Run the function creating a new Series object
    y22 = isf.quantile(x2, q, )
    # Run the NP function on the Series data
    expected_data = np.quantile(x2.data, q, axis=1, ).reshape(-1, len(q), )
    #
    assert np.all(y21.data == expected_data)
    assert np.all(y22.data == expected_data)


