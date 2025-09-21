
#[

import importlib.metadata as _md
import re as _re
import warnings as _wa

_BANNER = {
    "de": (
        "\n\n"
        "=================================================================\n"
        " You are using IrisPie Developer Edition.\n"
        " This edition is intended for development and testing only.\n"
        " It may be used solely by registered developers.\n"
        " Contact info@ogresearch.com for more information.\n"
        "=================================================================\n\n"
    ),
    "ce": (
        "\n\n"
        "=================================================================\n"
        " You are using IrisPie Community Edition.\n"
        " Free for personal, educational, and non-commercial use only.\n"
        " Registration is required for commercial or institutional use.\n"
        " Contact info@ogresearch.com for more information.\n"
        "=================================================================\n\n"
    ),
    "re": (
        "\n\n"
        "=================================================================\n"
        " You are using IrisPie Registered Edition.\n"
        " Licensed for use by organizations that have completed\n"
        " registration with OGResearch.\n"
        " Internal use and redistribution within the organization\n"
        " are permitted. External redistribution is prohibited.\n"
        " Contact info@ogresearch.com for more information.\n"
        "=================================================================\n\n"
    ),
    "pe": (
        "\n\n"
        "=================================================================\n"
        " You are using IrisPie Private Edition.\n"
        " This edition may only be used internally by staff members\n"
        " of OGResearch or by approved contractors and affiliated\n"
        " institutions.\n"
        "=================================================================\n\n"
    ),
}

distribution_generator = (
    i for i in _md.distributions()
    if _re.match("irispie-[dcrp]e$", i.name)
)
distribution = next(distribution_generator, None, )

if not distribution:
    raise Exception("Cannot determine the irispie distribution", )

edition = distribution.name[-2:]
version = distribution.version + "-" + edition
__version__ = version
__doc__ = distribution.metadata["description"]

_wa.warn(_BANNER[edition], UserWarning, )

#]


#[

def print_readme():
    print(__doc__)

def min_version_required(
    min_version_string: str,
):
    """
    """
    current_version = _convert_version(__version__, )
    minimum_version = _convert_version(min_version_string, )
    if current_version < minimum_version:
        raise Exception(
            f"Current version of irispie ({__version__}) is less than the minimum version required ({min_version_string})"
        )

min_irispie_version_required = min_version_required

def _convert_version(version_str: str) -> tuple[int, ...]:
    return tuple(int(s) for s in version_str.split("."))

#]


from .dates import *
from .dates import __all__ as dates_all

from .series import *
from .series import __all__ as series_all

from .ez_plotly import *
from .ez_plotly import __all__ as ez_plotly_all

from .databoxes import *
from .databoxes import __all__ as databoxes_all

from .dataslates import *
from .dataslates import __all__ as dataslates_all

from .chartpacks.main import *
from .chartpacks.main import __all__ as chartpacks_all

from .sources import *
from .sources import __all__ as sources_all

from .simultaneous import *
from .simultaneous import __all__ as simultaneous_all

from .stackers import *
from .stackers import __all__ as stackers_all

from .fords import *
from .fords import __all__ as fords_all

from .red_vars import *
from .red_vars import __all__ as red_vars_all

from .quantities import *
from .quantities import __all__ as quantities_all

from .equations import *
from .equations import __all__ as equations_all

from .sequentials import *
from .sequentials import __all__ as sequentials_all

from .explanatories import *
from .explanatories import __all__ as explanatories_all

from .plans import *
from .plans import __all__ as plans_all

from .namings import *
from .namings import __all__ as namings_all

from .file_io import *
from .file_io import __all__ as file_io_all

from .portables import *
from .portables import __all__ as portables_all

from .progress_bars import *
from .progress_bars import __all__ as progress_bars_all


__all__ = (
    *dates_all,
    *series_all,
    *ez_plotly_all,
    *databoxes_all,
    *dataslates_all,
    *sources_all,
    *simultaneous_all,
    *stackers_all,
    *fords_all,
    *red_vars_all,
    *quantities_all,
    *equations_all,
    *sequentials_all,
    *explanatories_all,
    *plans_all,
    *namings_all,
    *file_io_all,
    *portables_all,
    *progress_bars_all,
    "min_version_required",
    "min_irispie_version_required",
    "__version__",
)

