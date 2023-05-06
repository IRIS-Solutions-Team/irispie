"""
"""


#[
from __future__ import annotations

import numpy as np_
#]


FUNCTION_ADAPTATIONS = [
    "log", "exp", "sqrt", "maximum", "minimum"
]


for n in FUNCTION_ADAPTATIONS:
    exec(
        f"def {n}(x, *args, **kwargs): "
        f"return x._{n}_(*args, **kwargs) if hasattr(x, '_{n}_') else np_.{n}(x, *args, **kwargs)"
    )


def add_function_adaptations_to_custom_functions(context: dict | None) -> dict:
    """
    """
    #[
    context = context if context else {}
    for n in FUNCTION_ADAPTATIONS:
        context[n] = globals()[n]
    return context
    #]

