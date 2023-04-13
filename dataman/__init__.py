"""
Data management
"""


from ..dataman.dates import *
from ..dataman.dates import __all__ as dates_all

from ..dataman.databanks import *
from ..dataman.databanks import __all__ as databanks_all

from ..dataman.series import *
from ..dataman.series import __all__ as series_all

from ..dataman.plotly import *
from ..dataman.plotly import __all__ as plotly_all

__all__ = dates_all + databanks_all + series_all + plotly_all


