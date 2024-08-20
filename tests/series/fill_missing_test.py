
import irispie as ir
import random as rn
import functools as ft


start = ir.qq(2020,1)
end = ir.qq(2025,4)

start_missing = ir.qq(2021,4)
end_missing = ir.qq(2022,3)

x = ir.Series(periods=start>>end, func=ft.partial(rn.uniform, 1, 2), )

x_missing = x.copy()
x_missing[start_missing>>end_missing] = None


def test_fill_previous():
    x_filled = ir.fill_missing(x_missing, method="previous", )
    assert x_filled.any_missing(start>>end) == False


def test_fill_next():
    x_filled = ir.fill_missing(x_missing, method="next", )
    assert x_filled.any_missing(start>>end) == False


def test_fill_nearest():
    x_filled = ir.fill_missing(x_missing, method="nearest", )
    assert x_filled.any_missing(start>>end) == False


def test_fill_constant():
    x_filled = ir.fill_missing(x_missing, method="constant", method_args=100, )
    assert x_filled.any_missing(start>>end) == False


def test_fill_log_linear():
    x_filled = ir.fill_missing(x_missing, method="log_linear", )
    assert x_filled.any_missing(start>>end) == False


