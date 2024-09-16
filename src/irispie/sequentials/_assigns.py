"""
"""


#[

from __future__ import annotations

import numpy as _np
import documark as _dm

from .. import quantities as _quantities
from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants
from ..databoxes.main import (Databox, )

#]


class Inlay:
    """
    """
    #[

    @_dm.reference(category="parameters", )
    def assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
················································································

==Assign model parameters==

Assigns parameters to a `Sequential` model. The method can assign parameters
from individual arguments, from a `Databox`, or from a `dict`.


### Assigning individual parameters ###

```
self.assign(
    name_one=value_one,
    name_two=value_two,
    # etc
)
```


### Assigning parameters from a `Databox` or a `dict` ###

```
self.assign(databox, )
```


### Input arguments ###


???+ input "self"
    `Sequential` model whose parameters will be assigned.


???+ input "name_one, name_two, ..."
    Names of the parameters to assign.


???+ input "value_one, value_two, ..."
    Values to assign to `name_one`, `name_two`, etc.


???+ input "databox"
    `Databox` or `dict` from which the parameters will be extracted and
    assigned. Any names in the `Databox` or `dict` that are not model
    parameters will be ignored.


### Returns ###


???+ returns "None"
    This method modifies `self` in-place and does not return a value.


················································································
        """
        kwargs_to_assign = {}
        for a in args:
            kwargs_to_assign.update(a, )
        kwargs_to_assign.update(kwargs, )
        #
        kwargs_to_assign = {
            k: v
            for k, v in kwargs_to_assign.items()
            if k in self._invariant.parameter_names
        }
        #
        for v in self._variants:
            v.assign_from_dict_like(kwargs_to_assign, )

    @_dm.reference(category="parameters", )
    def check_missing_parameters(self, /, ) -> Databox:
        r"""
................................................................................

==Check for missing parameters==

Raises an error if any of the model parameters are missing values.

    self.check_missing_parameters()


### Input arguments ###


???+ input "self"
    `Sequential` model to check for missing parameters.


### Returns ###


Returns no value; raises an error if any parameters are missing, and prints the
list of missing parameter names.

................................................................................
        """
        parameters = self.get_parameters()
        missing = tuple(
            k for k, v in parameters.items()
            if _np.isnan(_np.array(v, dtype=float)).any()
        )
        if missing:
            message = ("Missing parameters: ", ) + missing
            raise _wrongdoings.IrisPieCritical(message, )

    @_dm.reference(category="parameters", )
    def get_parameters(
        self,
        /,
        unpack_singleton: bool = True,
    ) -> Databox:
        r"""
................................................................................

==Get model parameters==

Returns a `Databox` with the parameter values currently assigned within a
`Sequential` model.

    parameters = self.get_parameters(*, unpack_singleton=True, )


### Input arguments ###


???+ input "self"
    `Sequential` model whose parameters will be retrieved.

???+ input "unpack_singleton"
    If `True`, the method will unpack the parameters values for models with a
    single parameter variant.


### Returns ###


???+ return "parameters"
    `Databox` with the model parameters.

................................................................................
        """
        parameter_names = self._invariant.parameter_names
        parameters = {
            n: [ v.parameters[n] for v in self._variants ]
            for n in parameter_names
        }
        parameters = _has_variants.unpack_singleton_in_dict(
            parameters,
            self.is_singleton,
            unpack_singleton=unpack_singleton,
        )
        return Databox.from_dict(parameters, )

    #]

