"""
Calculate first derivatives of a custom function using finite approximation
"""


#[
from __future__ import annotations

from typing import (Callable, )
import numpy as np_
import copy as co_

from ..aldi import (differentiators as ad_, )
#]


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Exposure
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


def finite_differentiator(func: Callable) -> Callable:
    """
    Decorate a custom function for finite differentiation
    """
    def wrapper(*args):
        return _calculate_finite_derivatives(func, *args)
    return wrapper


#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
# Implementation
#••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••


_RELATIVE_FINITE_DIFF_STEP = 1e-6


def _calculate_finite_derivatives(
    func: Callable,
    /,
    *args,
) -> Callable:
    """
    Total finite differentiation of a custom function
    """
    arg_values = _collect_arg_values(*args, )
    new_value = func(*arg_values, )
    arg_diffs = _collect_arg_diffs(*args, )
    new_diff = sum(
        _partial_times_inner(func, k, arg_values, arg_diffs, )
        for k in range(len(args, ), )
    )
    return ad_.Atom.no_context(new_value, new_diff, False, )


def _partial_times_inner(func, k, arg_values, arg_diffs, ):
    """
    For f(..., g(x), ...), evaluate diff of f wrt the k-th argument times dg/dx
    """
    return (
        _partial_two_sided_derivative(func, k, arg_values, ) * arg_diffs[k]
        if arg_diffs[k] is not None else 0
    )


def _partial_two_sided_derivative(func, k, arg_values, ):
    """
    For f(..., x, ...), evaluate diff of f wrt the k-th argument
    """
    epsilon = _get_epsilon(arg_values[k])
    arg_values_plus = _plus_epsilon(arg_values, k, epsilon, )
    arg_values_minus = _plus_epsilon(arg_values, k, -epsilon, )
    return (func(*arg_values_plus) - func(*arg_values_minus)) / (2 * epsilon)


def _plus_epsilon(arg_values, k, epsilon, ):
    """
    Create a copy of function arguments and increase k-th argument by epsilon
    """
    arg_values_plus = co_.deepcopy(arg_values)
    arg_values_plus[k] += epsilon
    return arg_values_plus


def _get_epsilon(value, ):
    """
    Calculate the differentiation step based on the value around which we differentiate
    """
    base = np_.maximum(abs(value), 1)
    return base * _RELATIVE_FINITE_DIFF_STEP


def _collect_arg_values(*args, ):
    """
    Collect the values of input arguments, both Atom values and primitives
    """
    return [
        co_.deepcopy(a.value) if hasattr(a, "_is_atom", ) else a
        for a in args
    ]


def _collect_arg_diffs(*args, ):
    """
    Collect the diffs of input arguments for Atoms or Nones for primitives
    """
    return [
        co_.deepcopy(a.diff) if hasattr(a, "_is_atom", ) else None
        for a in args
    ]


