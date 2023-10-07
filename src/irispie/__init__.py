"""
------------------------------------------------------------


Iris Pie
=========

#### A Python package for macroeconomic modelling ####


------------------------------------------------------------
"""

import importlib.metadata as _md

from .dates import *
from .dates import __all__ as dates_all

from .series import *
from .series import __all__ as series_all

from .databoxes.main import *
from .databoxes.main import __all__ as databoxes_all

from .dataslates import *
from .dataslates import __all__ as dataslates_all

from .models import *
from .models import __all__ as models_all

from .quantities import *
from .quantities import __all__ as quantities_all

from .equations import *
from .equations import __all__ as equations_all

from .sequentials import *
from .sequentials import __all__ as sequentials_all

from .explanatories import *
from .explanatories import __all__ as explanatories_all

from .plans.main import *
from .plans.main import __all__ as plans_all

from .namings import *
from .namings import __all__ as namings_all


__version__ = _md.version(__name__)


#[
def min_irispie_version_required(
    minimum_version_str: str,
):
    """
    """
    current_version = _convert_version(__version__, )
    minimum_version = _convert_version(minimum_version_str, )
    if current_version < minimum_version:
        raise Exception(
            f"Current version of irispie ({__version__}) is less than minimum version required ({minimum_version_str})"
        )


def _convert_version(version_str: str) -> tuple[int, ...]:
    return tuple(int(s) for s in version_str.split("."))
#]


__all__ = (
    *dates_all,
    *series_all,
    *databoxes_all,
    *models_all,
    *quantities_all,
    *equations_all,
    *sequentials_all,
    *explanatories_all,
    *plans_all,
    *namings_all,
    "min_irispie_version_required",
    "__version__",
)

