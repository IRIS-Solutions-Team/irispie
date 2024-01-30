"""
Sequential models
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, Iterator, )
from typing import (Self, Any, )
import numpy as _np
import copy as _co
import os as _os

from .. import equations as _equations
from .. import quantities as _quantities
from .. import sources as _sources
from .. import pages as _pages
from ..incidences import main as _incidences
from ..incidences import blazer as _blazer
from ..explanatories import main as _explanatories
from .. import has_variants as _has_variants

from . import _invariants as _invariants
from . import _variants as _variants
from . import _simulate as _simulate
from . import _assigns as _assigns
from . import _get as _get

#]


__all__ = (
    "Sequential",
)


@_pages.reference(
    path=("structural_models", "sequential_models", "reference.md", ),
    categories={
        "constructor": "Creating new `Sequential` models",
        "property": None,
        "simulation": "Simulating `Sequential` models",
        "parameters": "Manipulating `Sequential` model parameters",
        "information": "Information about `Sequential` models",
        "manipulation": "Manipulating `Sequential` models",
    },
)
class Sequential(
    _simulate.Inlay,
    _assigns.Inlay,
    _get.Inlay,
    #
    _has_variants.HasVariantsMixin,
):
    """
......................................................................

`Sequential` model objects
============================

......................................................................
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    def __init__(
        self,
        /,
    ) -> None:
        """
        """
        self._invariant = None
        self._variants = []

    @_pages.reference(category="manipulation", )
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
    def skeleton(
        klass,
        other,
        /,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = other._invariant
        return self

    @classmethod
    @_pages.reference(category="constructor", call_name="Sequential.from_file", )
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
    Dictionary supplying the values used in preparsing commands, and the
    definition of non-standard functions used in the equations.

???+ input "description"
    Desscription of the model specified as a text string.


### Returns ###


???+ returns "self"

    `Sequential` model object created from the `file_names`.

················································································
        """
        return _sources.from_file(klass, *args, **kwargs, )

    @classmethod
    @_pages.reference(category="constructor", call_name="Sequential.from_string",)
    def from_string(klass, *args, **kwargs, ) -> _sources.SourceMixinProtocol:
        """
················································································

==Create sequential model object from string==

```
self = Sequential.from_string(
    string,
    /,
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

See [`Sequential.from_file`](sequentialfrom_file) for return values.

················································································
        """
        return _sources.from_string(klass, *args, **kwargs, )

    @classmethod
    def from_equations(
        klass,
        equations: Iterable[_equations.Equation],
        /,
        quantities: Iterable[_quantities.Quantity] | None = None,
        **kwargs,
    ) -> Self:
        """
        """
        self = klass()
        self._invariant = _invariants.Invariant.from_equations(equations, quantities, **kwargs, )
        initial_variant = _variants.Variant(self._invariant.parameter_names, )
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
    @_pages.reference(category="property", )
    def all_names(self, /, ) -> tuple[str]:
        """==Names of all variables occurring in the model in order of appearance=="""
        return tuple(self._invariant.all_names)

    @property
    @_pages.reference(category="property", )
    def lhs_names(self, /, ) -> tuple[str]:
        """==Names of LHS variables in order of their equations=="""
        return tuple(self._invariant.lhs_names)

    @property
    @_pages.reference(category="property", )
    def residual_names(self, /, ) -> tuple[str]:
        """==Names of residuals in order of their equations=="""
        return tuple(self._invariant.residual_names)

    @property
    @_pages.reference(category="property", )
    def rhs_only_names(self, /, ) -> tuple[str]:
        """==Names of variables appearing only on the RHS of equations=="""
        exclude_names = self._invariant.lhs_names + self._invariant.residual_names + self._invariant.parameter_names
        return tuple(
            n for n in self._invariant.all_names
            if n not in exclude_names
        )

    @property
    @_pages.reference(category="property", )
    def parameter_names(self, /, ) -> tuple[str]:
        """==Names of model parameters=="""
        return tuple(self._invariant.parameter_names)

    @property
    @_pages.reference(category="property", )
    def identity_index(self, /, ) -> tuple[int]:
        """==Indexes of identity equations=="""
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if x.is_identity
        )

    @property
    @_pages.reference(category="property", )
    def nonidentity_index(self, /, ) -> tuple[int]:
        """==Indexes of nonidentity equations=="""
        return tuple(
            i for i, x in enumerate(self._invariant.explanatories)
            if not x.is_identity
        )

    @property
    @_pages.reference(category="property", )
    def equation_strings(self, /, ) -> tuple[_equations.Equation]:
        """==Equation strings in order of appearance=="""
        return tuple(
            x.equation.human
            for x in self._invariant.explanatories
        )

    @property
    @_pages.reference(category="property", )
    def lhs_quantities(self, /, ) -> tuple[_quantities.Quantity]:
        """==LHS quantities in order of appearance=="""
        lhs_names = self._invariant.lhs_names
        kind = _quantities.QuantityKind.LHS_VARIABLE
        logly = False
        return tuple(
            _quantities.Quantity(qid, name, kind, logly, desc, )
            for (qid, name), desc in zip(enumerate(self._invariant.all_names), self.descriptions)
            if name in lhs_names
        )

    @property
    @_pages.reference(category="property", )
    def num_equations(self, /, ) -> int:
        """==Number of equations=="""
        return self._invariant.num_equations

    @property
    def descriptions(self, /, ) -> tuple[str]:
        """==Descriptions of equations in order of appearance=="""
        return tuple(
            x.equation.description
            for x in self._invariant.explanatories
        )

    @property
    @_pages.reference(category="property", )
    def incidence_matrix(self, /, ) -> _np.ndarray:
        """==Incidence matrix with equations in rows and LHS quantities in columns=="""
        def _shift_test(tok: _incidences.Token) -> bool:
            return tok.shift == 0
        return _equations.create_incidence_matrix(
            self.equations,
            self.lhs_quantities,
            shift_test=_shift_test,
        )

    @property
    @_pages.reference(category="property", )
    def min_shift(self, /, ) -> int:
        """==Maximum lag occurring on the RHS of equations=="""
        return min(
            x.min_shift
            for x in self._invariant.explanatories
        )

    @property
    @_pages.reference(category="property", )
    def max_shift(self, /, ) -> int:
        """==Maximum lead occurring on the RHS of equations=="""
        return max(
            x.max_shift
            for x in self._invariant.explanatories
        )

    @property
    @_pages.reference(category="property", )
    def is_sequential(
        self,
        /,
    ) -> bool:
        """==`True` if the model equations are ordered sequentially=="""
        return _blazer.is_sequential(self.incidence_matrix, )

    @property
    def equations(self, /, ) -> tuple[_equations.Equation]:
        return tuple(
            x.equation
            for x in self._invariant.explanatories
        )

    #
    #  Public methods
    #

    @_pages.reference(category="manipulation", )
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

    @_pages.reference(category="manipulation", )
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

    @_pages.reference(category="information", )
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

    @_pages.reference(category="information", )
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
            f"{self.__class__.__name__} model",
            f"Description: \"{self.get_description()}\"",
            f"|",
            f"|--Num of variants: {self.num_variants}",
            f"|--Num of equations: {self.num_equations}",
            f"|--Num of [nonidentities, identities]: [{len(self.nonidentity_index)}, {len(self.identity_index)}]",
            f"|--Num of parameters: {len(self.parameter_names)}",
            f"|--Num of rhs-only variables: {len(self.rhs_only_names)}",
            f"|--Time shifts [min, max]: [{self.min_shift:+g}, {self.max_shift:+g}]",
            f" ",
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

    #
    # ===== Implement SlatableProtocol =====
    #

    @_pages.reference(category="information", )
    def get_min_max_shifts(self, /, ) -> tuple[int, int]:
        """
················································································

==Get minimum and maximum shifts==

```
min_shift, max_shift = self.get_min_max_shifts()
```

Get the minimum shift (i.e., the maximum lag) and the maximum shift (i.e.,
the maximum lead) among all variables occuring in the model equations.


### Input arguments ###


???+ input "self"

    `Sequential` model object whose minimum and maximum shifts will be
    returned.


### Returns ###


???+ returns "min_shift"

    Minimum shift (i.e., the maximum lag).

???+ returns "max_shift"

    Maximum shift (i.e., the maximum lead).

················································································
        """
        return self.min_shift, self.max_shift

    @_pages.reference(category="information", )
    def get_databox_names(self, /, ) -> tuple[str]:
        """
················································································

==Get list of names that are extracted from databox for simulation==

```
names = self.get_databox_names()
```


### Input arguments ###


???+ input "self"

    `Sequential` model object whose databox names will be returned.


### Returns ###


???+ returns "names"

    List of names that are extracted from databox when the model is simulated.

················································································
        """
        return tuple(self._invariant.all_names)

    def get_fallbacks(self, /, ) -> dict[str, Any]:
        """
        """
        return { n: 0 for n in tuple(self._invariant.residual_names) }

    def get_overwrites(self, /, ) -> dict[str, Any]:
        """
        """
        return self.get_parameters()

    def get_scalar_names(self, /, ) -> tuple[str]:
        """
        """
        return tuple(self.parameter_names, )

    def create_qid_to_logly(self, /, ) -> dict[str, bool]:
        """
        """
        return {}

    #
    # ===== Implement PlannableSimulateProtocol =====
    #

    @property
    def simulate_can_be_exogenized(self, /, ) -> tuple[str, ...]:
        return tuple(set(
            i.lhs_name for i in self._invariant.explanatories
            if not i.is_identity
        ))

    @property
    def simulate_can_be_endogenized(self, /, ) -> tuple[str, ...]:
        return tuple(set(
            i.residual_name for i in self._invariant.explanatories
            if not i.is_identity
        ))

    simulate_can_be_anticipated = ()
    simulate_can_be_when_data = simulate_can_be_exogenized

    #]

