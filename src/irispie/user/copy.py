"""
Deep copy mixin
"""


#[
from __future__ import annotations
import copy as _cp
from typing import (Self, )
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
