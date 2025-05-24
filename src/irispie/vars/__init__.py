"""
Vector autoregression (VAR) module
"""

from .red_vars import *
from .red_vars import __all__ as red_vars_all

from .struct_vars import *
from .struct_vars import __all__ as struct_vars_all

from .prior_obs import *
from .prior_obs import __all__ as prior_obs_all

__all__ = (
    *red_vars_all,
    *struct_vars_all,
    *prior_obs_all,
)

