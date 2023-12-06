"""
"""


#[
from __future__ import annotations

from typing import (Self, )

from . import main as _databoxes
#]


class Dotter:
    """
    """
    #[

    @classmethod
    def from_databox(
        klass,
        db: _databoxes.Databox,
    ) -> Self:
        """
        """
        self = klass()
        self.__dict__.update(db, )
        return self

    def to_databox(
        self,
        db: _databoxes.Databox,
    ) -> None:
        """
        """
        db.update(self.__dict__, )

    #]


class DotterMixin:
    """
    """
    #[

    def __enter__(
        self,
    ) -> Self:
        """
        """
        new_dotter = Dotter.from_databox(self, )
        self._dotters.append(new_dotter, )
        return new_dotter


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        """
        current_dotter = self._dotters.pop()
        current_dotter.to_databox(self, )

    #]

