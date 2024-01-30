"""
"""


#[
from __future__ import annotations

import numpy as _np

from .. import quantities as _quantities
from .. import pages as _pages
from .. import wrongdoings as _wrongdoings
from ..databoxes import main as _databoxes
#]


class Inlay:
    """
    """
    #[

    @_pages.reference(category="parameters", )
    def assign(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
················································································

==Assign model parameters==

```
self.assign(
    name_one=value_one,
    name_two=value_two,
    # etc
)
```

```
self.assign(databox, )
```

...


### Input arguments ###


???+ input "self"

    `Sequential` model whose parameters will be assigned.


???+ input "name_one"

    Name of a parameter to assign.


???+ input "value_one"

    Value to assign to `name_one`.

etc...

???+ input "databox"

    `Databox` or `dict` from which the parameters will be extracted and
    assigned.

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

    def check_missing_parameters(self, /, ) -> _databoxes.Databox:
        """
        """
        parameters = self.get_parameters()
        missing = tuple(
            k for k, v in parameters.items()
            if _np.isnan(_np.array(v, dtype=float)).any()
        )
        if missing:
            raise _wrongdoings.IrisPieCritical(("Missing parameters", ) + missing)

    def get_parameters(self, /, ) -> _databoxes.Databox:
        """
        """
        parameter_names = self._invariant.parameter_names
        if self.is_singleton:
            return _databoxes.Databox(**{
                n: self._variants[0].parameters[n]
                for n in parameter_names
            })
        else:
            return _databoxes.Databox(**{
                n: [ v.parameters[n] for v in self._variants ]
                for n in parameter_names
            })

    #]

