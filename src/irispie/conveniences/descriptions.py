"""
Description mixin
"""


#[
from __future__ import annotations

import documark as _dm
#]


class DescriptionMixin:
    """
    """
    #[

    @_dm.reference(category="information", )
    def get_description(self, /, ) -> str:
        r"""
................................................................................


==Get the description attached an Iris Pie object==

    description = self.get_description()

### Input arguments ###

???+ input "self"
    An Iris Pie object from which to get the description.


### Returns ###

???+ returns "description"
    The description attached to the Iris Pie object.


................................................................................
        """
        return str(self.__description__ or "")

    @_dm.reference(category="information", )
    def set_description(self, description: str, /, ) -> None:
        r"""
................................................................................


==Set the description for an Iris Pie object==

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

