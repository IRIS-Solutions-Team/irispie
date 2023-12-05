"""
"""


#[
from __future__ import annotations

import numpy as _np
import scipy as _sp
#]


_ELEMENTWISE_FUNCTIONS = {
    "log": _np.log,
    "exp": _np.exp,
    "sqrt": _np.sqrt,
    "logistic": _sp.special.expit,
    "maximum": _np.maximum,
    "minimum": _np.minimum,
}


for n in _ELEMENTWISE_FUNCTIONS.keys():
    exec(
        f"def {n}(x, *args, **kwargs):\n"
        f"    if hasattr(x, '{n}'):\n"
        f"        return x.{n}(*args, **kwargs, )\n"
        f"    else:\n"
        f"        return _ELEMENTWISE_FUNCTIONS['{n}'](x, *args, **kwargs, )\n"
    )


def add_function_adaptations_to_context(context: dict | None) -> dict:
    """
    """
    #[
    context = context if context else {}
    for n in _ELEMENTWISE_FUNCTIONS.keys():
        context[n] = globals()[n]
    return context
    #]

