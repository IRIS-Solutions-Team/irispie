
from __future__ import annotations

import numpy

from .dataman import series

__all__ = series.underscore_functions

for n in __all__:
    exec(
f"""
def {n}(x, *args, **kwargs):
    return x._{n}_(*args, **kwargs) if hasattr(x, '_{n}_') else numpy.{n}(x, *args, **kwargs)
"""
    )

