
import pytest
import numpy as np
import irispie as ir

rng = np.random.default_rng(0)

num_periods = 10
start = ir.qq(2020, 1)
end = start + num_periods - 1 

x1 = rng.random(size=(num_periods, ), )
x2 = rng.random(size=(num_periods, 3, ), )
x3 = rng.random(size=(num_periods, 3, 2), )

y3 = x3.copy()
y3[-1] = np.nan


def test_from_start_and_array_1d():
    s = ir.Series.from_start_and_array(
        start=start,
        array=x1,
    )
    assert s.periods[0] == start
    assert len(s.periods) == num_periods
    assert s.data.shape == (num_periods, 1)
    assert s.data[0] == x1[0]
    assert s.data[-1] == x1[-1]


def test_from_start_and_array_2d():
    s = ir.Series.from_start_and_array(
        start=start,
        array=x2,
    )
    assert s.periods[0] == start
    assert len(s.periods) == num_periods
    assert s.data.shape == (num_periods, 3, )
    assert s.data[0, 0] == x2[0, 0]
    assert s.data[-1, -1] == x2[-1, -1]


def test_from_start_and_array_3d():
    s = ir.Series.from_start_and_array(
        start=start,
        array=x3,
    )
    assert s.periods[0] == start
    assert len(s.periods) == num_periods
    assert s.data.shape == (num_periods, 6, )
    assert s.data[0, 0] == x3[0, 0, 0]
    assert s.data[-1, -1] == x3[-1, -1, -1]


def test_from_start_and_array_3d_nan():
    s = ir.Series.from_start_and_array(
        start=start,
        array=y3,
    )
    assert s.periods[0] == start
    assert len(s.periods) == num_periods-1
    assert s.data.shape == (num_periods-1, 6, )
    assert s.data[0, 0] == x3[0, 0, 0]
    assert s.data[-1, -1] == x3[-2, -1, -1]


