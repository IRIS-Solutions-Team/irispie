"""
"""


#[
from __future__ import annotations

import numpy as np_

from ..aldi import (differentiators as ad_, )
#]


FUNCTION_ADAPTATIONS = ad_.FUNCTION_ADAPTATIONS


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

