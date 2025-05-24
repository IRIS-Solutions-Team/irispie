r"""
Sequential models
"""


#[

from __future__ import annotations

import numpy as _np
import copy as _co
import os as _os
import documark as _dm

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from ..incidences.main import Token
from ..incidences import main as _incidences
from ..incidences import blazer as _blazer
from ..explanatories import main as _explanatories
from .. import has_variants as _has_variants

from . import _simulate as _simulate
from . import _assigns as _assigns
from . import _get as _get
from . import _slatable_protocols as _slatable_protocols
from . import _plannable_protocols as _plannable_protocols
from ._invariants import Invariant
from ._variants import Variant

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Any
    from collections.abc import Iterable, Iterator
    from ..equations import Equation
    from ..quantities import Quantity

#]


__all__ = (
    "Sequential",
)


@_dm.reference(
    path=("structural_models", "sequential.md", ),
    categories={
        "constructor": "Creating new Sequential models",
        "property": None,
        "information": "Getting information about Sequential models",
        "simulation": "Simulating Sequential models",
        "parameters": "Manipulating Sequential model parameters",
        "manipulation": "Manipulating Sequential models",
    },
)
class Sequential(
    _simulate.Inlay,
    _assigns.Inlay,
    _get.Inlay,
    _slatable_protocols.Inlay,
    _plannable_protocols.Inlay,
    #
    _has_variants.Mixin,
):
    """
......................................................................

`Sequential` models
====================

`Sequential` models are models where the equations are simulated sequentially,
one data point at a time. The order of execution in simulations is either
period-by-period and equation-by-equation, or vice versa, equation-by-equation
and period-by-period.

......................................................................
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(
        self,
        invariant: Invariant | None = None,
        variants: Iterable[Variant] | None = None,
    ) -> None:
        r"""
        """
        self._invariant = Invariant()
        self._variants = list(variants) if variants else []

    @_dm.reference(category="manipulation", )
    def copy(self, /, ) -> Self:
        """
················································································

==Create a deep copy==

```
other = self.copy()
```


### Input arguments ###


???+ input "self"
    A `Sequential` model object to be copied.


### Returns ###


???+ returns "other"
    A deep copy of `self`.

......................................................................
        """
        return _co.deepcopy(self, )

    @classmethod
    @_dm.reference(category="constructor", call_name="Sequential.from_file", )
    def from_file(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        """
················································································

==Create new `Sequential` model object from source file or files==

```
self = Sequential.from_file(
    file_names,
    /,
    context=None,
    description="",
)
```

Read and parse one or more source files specified by `file_names` (a string
or a list of strings) with model source code, and create a `Sequential`
model object.


### Input arguments ###


???+ input "file_names"
    The name of the model source file from which the `Sequential` model object
    will be created, or a list of file names; if multiple file names are
    specified, they will all combined together in the given order.

???+ input "context"
    Dictionary supplying the values used in [preparsing](preparser.md) commands, and the
    definition of non-standard functions used in the equations.

???+ input "description"
    Description of the model specified as a text string.


### Returns ###


???+ returns "self"
    A new `Sequential` model object created from the `file_names`.

················································································
        """
        return _sources.from_file(klass, *args, **kwargs, )

    @classmethod
    @_dm.reference(category="constructor", call_name="Sequential.from_string",)
    def from_string(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        """
················································································

==Create sequential model object from string==

```
self = Sequential.from_string(
    string,
    /,
    *,
    context=None,
    description="",
)
```

Read and parse a text `string` with a model source code, and create a
`Sequential` model object. Otherwise, this function behaves the same way as
[`Sequential.from_file`](#sequentialfrom_file).


### Input arguments ###

???+ input "string"

    Text string from which the `Sequential` model object will be created.

See [`Sequential.from_file`](#sequentialfrom_file) for other input arguments.



### Returns ###

See [`Sequential.from_file`](#sequentialfrom_file) for return values.

················································································
        """
        return _sources.from_string(klass, *args, **kwargs, )

    @classmethod
    def from_equations(
        klass,
        equations: Iterable[Equation],
        /,
        quantities: Iterable[Quantity] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = Invariant.from_equations(equations, quantities, **kwargs, )
        initial_variant = Variant(self._invariant.parameter_names, )
        self._variants = [ initial_variant ]
        return self

    @classmethod
    def from_source(
        klass,
        source: _sources.ModelSource,
        /,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass.from_equations(
            source.dynamic_equations,
            quantities=source.quantities,
            **kwargs,
        )
        return self

    #
    #  Properties
    #

    @property
    @_dm.reference(category="property", )
    def all_names(self, /, ) -> tuple[str]:
        """==Names of all variables occurring in the model in order of appearance=="""
        return tuple(self._invariant.all_names)

    @property
    @_dm.reference(category="property", )
    def lhs_names(self, /, ) -> tuple[str]:
        """==Unique names of LHS variables in order of their first appearance in equations=="""
        return self._invariant.lhs_names

    @property
    @_dm.reference(category="property", )
    def lhs_names_in_equations(self, /, ) -> tuple[str]:
        """==Names of LHS variables in order of their appearance in equations=="""
        return tuple( x.lhs_name for x in self._invariant.explanatories )

    @property
    @_dm.reference(category="property", )
    def residual_names(self, /, ) -> tuple[str]:
        """==Unique names of residuals in order of their first appearance in  equations=="""
        return self._invariant.residual_names

    @property
    @_dm.reference(category="property", )
    def residual_names_in_equations(self, /, ) -> tuple[str]:
        """==Names of residuals in order of their appearance in  equations=="""
        return tuple( x.residual_name for x in self._invariant.explanatories )

    @property
    @_dm.reference(category="property", )
    def rhs_only_names(self, /, ) -> tuple[str]:
        """==Names of variables appearing only on the RHS of equations=="""
        exclude_names = self._invariant.lhs_names + self._invariant.residual_names + self._invariant.parameter_names
        return tuple(
            n for n in self._invariant.all_names
            if n not in exclude_names
        )

    @property
    @_dm.reference(category="property", )
    def parameter_names(self, /, ) -> tuple[str]:
        """==Names of model parameters=="""
        return tuple(self._invariant.parameter_names)

    @property
    @_dm.reference(category="property", )
    def identity_index(self, /, ) -> tuple[int]:
        """==Indexes of identity equations=="""
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if x.is_identity
        )

    @property
    @_dm.reference(category="property", )
    def nonidentity_index(self, /, ) -> tuple[int]:
        """==Indexes of nonidentity equations=="""
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if not x.is_identity
        )

    @property
    @_dm.reference(category="property", )
    def equation_strings(self, /, ) -> tuple[Equation]:
        """==Equation strings in order of appearance=="""
        return tuple(
            x.equation.human
            for x in self._invariant.explanatories
        )

    @property
    @_dm.reference(category="property", )
    def lhs_quantities(self, /, ) -> tuple[Quantity]:
        """==LHS quantities in order of appearance=="""
        lhs_names = self._invariant.lhs_names
        kind = QuantityKind.LHS_VARIABLE
        logly = False
        name_to_qid = self.create_name_to_qid()
        return tuple(
            Quantity(name_to_qid[name], name, kind, logly, self.descriptions[name_to_qid[name]], )
            for name in self.lhs_names
        )

    @property
    @_dm.reference(category="property", )
    def num_equations(self, /, ) -> int:
        """==Number of equations=="""
        return self._invariant.num_equations

    @property
    @_dm.reference(category="property", )
    def num_lhs_names(self, /, ) -> int:
        """==Number of unique LHS names=="""
        return self._invariant.num_lhs_names

    @property
    def descriptions(self, /, ) -> tuple[str]:
        """==Descriptions of equations in order of appearance=="""
        return tuple(
            x.equation.description
            for x in self._invariant.explanatories
        )

    @property
    @_dm.reference(category="property", )
    def incidence_matrix(self, /, ) -> _np.ndarray:
        """==Incidence matrix with equations in rows and LHS quantities in columns=="""
        num_lhs_names = len(self.lhs_names)
        def token_within_quantities(tok: Token, /, ) -> bool:
            return (
                tok.qid
                if tok.qid < num_lhs_names and tok.shift == 0
                else None
            )
            #
        return _equations.calculate_incidence_matrix(
            self.equations,
            len(self.lhs_names),
            token_within_quantities,
        )

    @property
    @_dm.reference(category="property", )
    def max_lag(self, /, ) -> int:
        """==Maximum lag occurring on the RHS of equations=="""
        return min(
            x.min_shift
            for x in self._invariant.explanatories
        ) if self._invariant.explanatories else 0

    @property
    def min_shift(self, /, ) -> int:
        return self.max_lag

    @property
    @_dm.reference(category="property", )
    def max_lead(self, /, ) -> int:
        """==Maximum lead occurring on the RHS of equations=="""
        return max(
            x.max_shift
            for x in self._invariant.explanatories
        ) if self._invariant.explanatories else 0

    @property
    def max_shift(self, /, ) -> int:
        return self.max_lead

    @property
    @_dm.reference(category="property", )
    def is_sequential(
        self,
        /,
    ) -> bool:
        """==`True` if the model equations are ordered sequentially=="""
        return (
            _blazer.is_sequential(self.incidence_matrix, )
            if self._invariant.explanatories else True
        )

    @property
    def equations(self, /, ) -> tuple[Equation]:
        return self._invariant.equations

    @_dm.reference(category="manipulation", )
    def reorder_equations(self, *args, **kwargs, ) -> None:
        """
......................................................................

==Reorder model equations==

```
self.reorder_equations(new_order, )
```

Reorder the model equations within `self` according to the `new_order` of
equation indexes.


### Input arguments ###


???+ input "self"

    `Sequential` model object whose equations will be reordered.

???+ input "new_order"

    New order of model equations specified as a list of equation indexes
    (integers starting from 0).

......................................................................
        """
        self._invariant.reorder_equations(*args, **kwargs, )

    @_dm.reference(category="manipulation", )
    def sequentialize(
        self,
        /,
    ) -> tuple[int, ...]:
        """
......................................................................

==Reorder the model equations so that they can be solved sequentially==

```
eids_reordered = self.sequentialize()
```

Reorder the model equations within `self` so that they can be solved
sequentially. The reordered equation indexes are returned as a tuple.


### Input arguments ###


???+ input "self"

    `Sequential` model object whose equations will be reordered sequentially.


### Returns ###


???+ returns "eids_reordered"

    Tuple of equation indexes (integers starting from 0) specifying the
    new order of equations.

......................................................................
        """
        if self.is_sequential:
            return tuple(range(self.num_equations))
        eids_reordered = _blazer.sequentialize_strictly(self.incidence_matrix, )
        self.reorder_equations(eids_reordered, )
        return tuple(eids_reordered)

    def iter_equations(self, /, ) -> Iterator[_explanatories.Explanatory]:
        """
        """
        yield from self._invariant.explanatories

    iter_explanatories = iter_equations

    @_dm.reference(category="information", )
    def get_description(self, /, ) -> str:
        """
················································································

==Get model description text==

```
description = self.get_description()
```

### Input arguments ###


???+ input "self"

    `Sequential` model object whose description will be returned.


### Returns ###


???+ returns " "

    Description of `self`.

················································································
        """
        return self._invariant.get_description()

    @_dm.reference(category="information", )
    def set_description(self, *args, **kwargs, ) -> None:
        """
················································································

==Set model description text==

```
self.set_description(description, )
```

...


### Input arguments ###


???+ input "self"

    `Sequential` model object whose description will be set.


???+ input "description"

    New description of `self`.

················································································
        """
        self._invariant.set_description(*args, **kwargs, )

    def create_qid_to_name(self, *args, **kwargs, ) -> dict[int, str]:
        """
        """
        return self._invariant.create_qid_to_name(*args, **kwargs, )

    def create_name_to_qid(self, *args, **kwargs, ) -> dict[str, int]:
        """
        """
        return self._invariant.create_name_to_qid(*args, **kwargs, )

    def __repr__(self, /, ) -> str:
        """
        """
        indented = " " * 4
        return "\n".join((
            f"",
            f"<{self.__class__.__name__} at {id(self):#x}>",
            f"[Description: \"{self.get_description()}\"]",
            f"[Num variants: {self.num_variants}]",
            f"[Num equations, nonidentities, identities: {self.num_equations}, {len(self.nonidentity_index)}, {len(self.identity_index)}]",
            f"[Num parameters: {len(self.parameter_names)}]",
            f"[Num rhs-only variables: {len(self.rhs_only_names)}]",
            f"[Max lag, lead: t{self.min_shift:+g}, t{self.max_shift:+g}]",
            f"",
        ))

    def __str__(self, /, ) -> str:
        """
        """
        return repr(self, )

    def __getitem__(
        self,
        request: int,
        /,
    ) -> Self:
        """
        """
        return self.get_variant(request, )

    #]

