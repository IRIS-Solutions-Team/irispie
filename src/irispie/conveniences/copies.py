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
    def copy(self, /, ) -> Self:
        """
        """
        return _cp.deepcopy(self)
    #]

