"""
"""


#[
from __future__ import annotations

from .. import quantities as _quantities
from .. import pages as _pages
from ..databoxes import main as _databoxes
#]


class AssignInlay:
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
        for v in self._variants:
            v.assign(*args, **kwargs, )

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

