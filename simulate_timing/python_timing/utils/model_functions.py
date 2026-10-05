# Code by OGResearch
"""
Copy of modules/codes/python/general/utils/model_functions.py, so that this
folder runs without ogi. The functions a model equation may call that
irispie does not provide: erf, sabs, smax0, smax0_, smin0.
"""
import numpy as np
import scipy.special

__all__ = ["model_context_functions", "sabs", "smax0", "smax0_", "smin0"]


def sabs(b, c):
    """
    Smooth approximation of `abs(b)`, with `c` setting the width of the kink.

    MATLAB name: sabs

    Parameters
    ----------
    b
        The value whose absolute value is approximated. Scalar or array.
    c
        Smoothing parameter. The approximation is exact as `c` goes to zero
        and flattens as `c` grows.

    Returns
    -------
    The smoothed absolute value, `b*b / sqrt(b*b + c*c)`.
    """
    return b * b / np.sqrt(b * b + c * c)


def smax0(b, c):
    """
    Smooth approximation of `max(b, 0)`, with `c` setting the kink width.

    MATLAB name: smax0

    Parameters
    ----------
    b
        The value floored at zero. Scalar or array.
    c
        Smoothing parameter, passed on to `sabs`.

    Returns
    -------
    `0.5 * (b + sabs(b, c))`.
    """
    return 0.5 * (b + sabs(b, c))


def smin0(b, c):
    """
    Smooth approximation of `min(b, 0)`, with `c` setting the kink width.

    MATLAB name: smin0

    Parameters
    ----------
    b
        The value capped at zero. Scalar or array.
    c
        Smoothing parameter, passed on to `sabs`.

    Returns
    -------
    `0.5 * (b - sabs(b, c))`.
    """
    return 0.5 * (b - sabs(b, c))


def smax0_(b, c):
    """
    Leaky `max(b, 0)`: `b` where it is non-negative, `c * b` where it is not.

    MATLAB name: smax0_

    Unlike `smax0` this is piecewise linear rather than smooth -- the kink at
    zero is real, and `c` is the slope below it rather than a smoothing
    width. `gpm` uses it for the soft interest rate floor, with `c = 0.01`.

    Parameters
    ----------
    b
        The value floored at zero. Scalar or array.
    c
        Slope applied to the negative part.

    Returns
    -------
    `b` where `b >= 0`, `c * b` elsewhere. An array argument returns an array;
    a scalar returns a scalar, as MATLAB's logical-index assignment does.
    """
    return np.where(b < 0, c * b, b)


def model_context_functions() -> dict:
    """
    The functions a model source may call that irispie does not provide.

    Handed to `ir.Simultaneous.from_file` / `from_string` as (part of) the
    `context`, which is where irispie looks up any name in an equation that
    is neither a quantity nor one of its own built-in functions.

    Returned as a fresh dict on every call so a caller can add its own
    context entries to it without affecting the next caller.

    Returns
    -------
    dict
        `{"erf": ..., "sabs": ..., "smax0": ..., "smax0_": ..., "smin0": ...}`.
    """
    return {
        "erf": scipy.special.erf,
        "sabs": sabs,
        "smax0": smax0,
        "smax0_": smax0_,
        "smin0": smin0,
    }
