"""
"""


#[
from __future__ import annotations

from typing import (NoReturn, Callable, )

from ..aldi import (differentiators as ad_, )
#]


_REL_FINITE_DIFF_STEP = 1e-6


def _finite_differentiator(
    func: Callable,
    /,
    *args,
) -> Callable:
    """
    Total finite differentiation of a custom function
    """
    y = func(*args)
    if not hasattr(y, "_is_atom"):
        return y
    new_value = y._value
    arg_values = _collect_arg_values(*args, )
    arg_diffs = _collect_arg_diffs(*args, )
    new_diff = sum(
        _partial_times_inner(func, k, arg_values, arg_diffs, )
        for k in range(len(args, ), )
    )
    return ad_.Atom.no_context(new_value, new_diff, False)


def _partial_times_inner(func, k, arg_values, arg_diffs, ):
    """
    For f(..., g(x), ...), evaluate diff of f w.r.t. to n-th argument times dg/dx
    """
    return (
        _partial_two_sided_derivative(func, k, arg_values, ) * arg_diffs[k]
        if arg_diffs[k] is not None else 0
    )


def _partial_two_sided_derivative(func, k, arg_values, ):
    """
    """
    epsilon = _get_epsilon(arg_values[k])
    arg_values_plus = _plus_epsilon(arg_values, k, epsilon, )
    arg_values_minus = _plus_epsilon(arg_values, k, -epsilon, )
    return (func(*arg_values_plus) - func(*arg_values_minus)) / (2 * epsilon)


def _plus_epsilon(arg_values, k, epsilon, ):
    arg_values_plus = arg_values[:]
    arg_values_plus[k] += epsilon
    return arg_values_plus


def _get_epsilon(value, ):
    """
    """
    return min(1, abs(_REL_FINITE_DIFF_STEP * value), )


def _collect_arg_values(*args, ):
    """
    """
    return [
        a._value if hasattr(a, "_is_atom", ) else a
        for a in args
    ]


def _collect_arg_diffs(*args, ):
    """
    """
    return [
        a._diff if hasattr(a, "_is_atom", ) else None
        for a in args
    ]


