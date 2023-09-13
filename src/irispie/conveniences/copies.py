"""
Deep copy mixin
"""


#[
from __future__ import annotations

from typing import (Self, )
import copy as _cp
#]


class CopyMixin:
    """
    """
    #[
    def precopy(self, /, ):
        pass

    def copy(self, /, ) -> Self:
        """
        """
        self.precopy()
        return _cp.deepcopy(self)
    #]

