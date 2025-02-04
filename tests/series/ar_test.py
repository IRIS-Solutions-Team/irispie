
import pytest
import random as rn
import numpy as np
import irispie as ir


rn.seed(0)

N = 100

def random_period():
    freq = rn.choices([1, 2, 4, 12], )[0]
    year = rn.randint(2000, 2050, )
    seg = rn.randint(1, freq, )
    return ir.Period.from_year_segment(freq, year, seg, )


def random_series():
    return ir.Series(
        start=random_period(),
        values=rn.gauss(),
        num_variants=rn.randint(1, 5, ),
    )


@pytest.mark.parametrize(
    ["x", "horizon", ], [(random_series(), rn.randint(4,10)) for _ in range(N, )],
)
def test_extrapolate(x: ir.Series, horizon: int, ) -> None:
    y = ir.extrapolate(x, (0.8, ), x.end+1 >> x.end+horizon, intercept=0, )
    x_end = x[x.end, :].flatten()
    y_end = y[y.end, :].flatten()
    assert np.allclose(x_end * 0.8**horizon, y_end, atol=1e-6, rtol=1e-6, )

