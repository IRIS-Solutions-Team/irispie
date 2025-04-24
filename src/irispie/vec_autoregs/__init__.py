"""
Vector autoregression (VAR) module
"""

from .red_vars import *
from .red_vars import __all__ as red_vars_all

from .str_vars import *
from .str_vars import __all__ as str_vars_all

from .prior_dummies import *
from .prior_dummies import __all__ as prior_dummies_all

__all__ = (
    *red_vars_all,
    *str_vars_all,
    *prior_dummies_all,
)

