r"""
Description mixin
"""


#[

from __future__ import annotations

from typing import Protocol
import documark as _dm

#]


class DescriptionableProtocol(Protocol, ):
    r"""
    """
    #[

    __description__: str | None = None

    #]


class DescriptionMixin:
    r"""
    """
    #[

    @_dm.reference(category="information", )
    def get_description(
        self: DescriptionableProtocol,
    ) -> str:
        r"""
................................................................................


==Get description attached to an object==

    description = self.get_description()


### Input arguments ###

???+ input "self"
    An object from which to get the description.


### Returns ###

???+ returns "description"
    The description attached to the object.


................................................................................
        """
        return str(self.__description__ or "")

    @_dm.reference(category="information", )
    def set_description(
            self: DescriptionableProtocol,
            description: str,
        ) -> None:
        r"""
................................................................................


==Set the description for an object==

    self.set_description(
        description,
    )


### Input arguments ###

???+ input "self"
    An Iris Pie object to which to attach the description.


???+ input "description"
    The description to attach to the Iris Pie object.


### Returns ###

This method modifies the Iris Pie object in place and returns `None`.


................................................................................
        """
        self.__description__ = str(description or "")

    #]

