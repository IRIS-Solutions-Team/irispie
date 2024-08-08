"""
Cells in simulation dataslates
"""


#[
from __future__ import annotations

from typing import (NamedTuple, Self, )
#]


class Cell(NamedTuple, ):
    """
    Cell in a simulation dataslate
    """
    #[

    qid: int
    column: int

    def shifted(
        self: Self,
        by: int,
    ) -> Self:
        """
        Create a new Cell with a shifted column
        """
        return Cell(self.qid, self.column+by, )

    #]


