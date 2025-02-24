"""
Methods and functions applied elementwise to Series values
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp
import functools as _ft
import textwrap as _tw

#]


_ONE_ARG_FUNCTION_DISPATCH = {
    "log": "_np.log",
    "log2": "_np.log2",
    "log10": "_np.log10",
    "log1p": "_np.log1p",
    "exp": "_np.exp",
    "exp2": "_np.exp2",
    "expm1": "_np.expm1",
    "sqrt": "_np.sqrt",
    "abs": "_np.abs",
    "sign": "_np.sign",
    "sin": "_np.sin",
    "cos": "_np.cos",
    "tan": "_np.tan",
    "asin": "_np.asin",
    "acos": "_np.acos",
    "atan": "_np.atan",
    "expit": "_sp.special.expit",
    "logistic": "_sp.special.expit",
    "erf": "_sp.special.erf",
    "erfinv": "_sp.special.erfinv",
    "erfc": "_sp.special.erfc",
    "erfcinv": "_sp.special.erfcinv",
    "normal_cdf": "_sp.stats.norm.cdf",
    "normal_pdf": "_sp.stats.norm.pdf",
}

_TWO_ARGS_FUNCTION_DISPATCH = {
    "round": "_np.round",
    "maximum": "_np.maximum",
    "minimum": "_np.minimum",
}

_FUNCTION_DISPATCH = {
    **_ONE_ARG_FUNCTION_DISPATCH,
    **_TWO_ARGS_FUNCTION_DISPATCH,
}

__all__ = list(_FUNCTION_DISPATCH.keys(), )


_METHOD_CODE = r"""
    def {k}(self, *args, **kwargs, ):
        self.data = {v}(self.data, *args, **kwargs, )
"""
class Inlay:
    for k, v in _FUNCTION_DISPATCH.items():
        code = _tw.dedent(_METHOD_CODE.format(k=k, v=v, ))
        exec(code, globals(), locals(), )


_FUNC_CODE = r"""
    def {k}(object, *args, **kwargs, ):
        if hasattr(object, '{k}'):
            new = object.copy()
            new.{k}(*args, **kwargs, )
            return new
        else:
            return {v}(object, *args, **kwargs, )
"""
for k, v in _FUNCTION_DISPATCH.items():
    code = _tw.dedent(_FUNC_CODE.format(k=k, v=v, ))
    exec(code, globals(), locals(), )


