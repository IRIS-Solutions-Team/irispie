
import pytest
import random as rn
import numpy as np
import irispie as ir


rn.seed(0)


def sum_round(x, tolerance=10, ):
    return np.round(x, tolerance).sum()


def random_series():
    return ir.Series(periods=ir.qq(2020,1)>>ir.qq(2025,4), func=rn.gauss, )


N = 10


@pytest.mark.parametrize(
    ["x"], [ (random_series(), ) for _ in range(N, ) ]
)
def test_hpf_additive(x, ):
    xt, xg = ir.hpf(x, )
    assert sum_round((xt + xg).get_data() - x.get_data()) == 0


@pytest.mark.parametrize(
    ["x", "level_value"], [(random_series(), rn.uniform(0, 1)) for _ in range(N, )]
)
def test_hpf_level(x, level_value, ):
    x = ir.Series(periods=ir.qq(2020,1)>>ir.qq(2025,4), func=rn.gauss, )
    level_date = x.end
    level = ir.Series(periods=level_date, values=level_value, )
    xt, xg = ir.hpf(x, level=level, )
    assert sum_round(xt[level_date] - level_value, 0) == 0


@pytest.mark.parametrize(
    ["x", "change_value"], [(random_series(), rn.uniform(0, 1)) for _ in range(N, )]
)
def test_hpf_change(x, change_value, ):
    x = ir.Series(periods=ir.qq(2020,1)>>ir.qq(2025,4), func=rn.gauss, )
    change_date = x.end
    change = ir.Series(periods=change_date, values=change_value, )
    xt, xg = ir.hpf(x, change=change, )
    diff_xt = ir.diff(xt)
    assert sum_round(diff_xt[change_date] - change_value, 0) == 0


