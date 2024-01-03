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

from .plotly_wrap import *
from .plotly_wrap import __all__ as plotly_wrap_all

from .rephrases import *
from .rephrases import __all__ as rephrases_all

from .databoxes.main import *
from .databoxes.main import __all__ as databoxes_all

from .chartpacks.main import *
from .chartpacks.main import __all__ as chartpacks_all

from .dataslates.main import *
from .dataslates.main import __all__ as dataslates_all

from .sources import *
from .sources import __all__ as sources_all

from .simultaneous import *
from .simultaneous import __all__ as simultaneous_all

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
    *plotly_wrap_all,
    *rephrases_all,
    *databoxes_all,
    *sources_all,
    *simultaneous_all,
    *quantities_all,
    *equations_all,
    *sequentials_all,
    *explanatories_all,
    *plans_all,
    *namings_all,
    "min_irispie_version_required",
    "__version__",
)

import dill as _di

def save(obj, filename, ):
    with open(filename, "wb+", ) as fid:
        _di.dump(obj, fid, )

def load(filename, ):
    with open(filename, "rb", ) as fid:
        return _di.load(fid, )


def _test_docs_():
    """
# Testing docs
    """
    pass


