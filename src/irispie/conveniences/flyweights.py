"""
Simple temporary flyweight implementation
"""


#[
from __future__ import annotations
#]


class FlyweightRegister:
    """
    """
    #[

    def __init__(self, flyweight_factory: type, ) -> None:
        """
        """
        self._flyweight_factory = flyweight_factory
        self._flyweights = {}

    def __call__(self, *args, ):
        """
        """
        key = args
        if key not in self._flyweights:
            self._flyweights[key] = self._flyweight_factory(*args, )
        return self._flyweights[key]

    #]

