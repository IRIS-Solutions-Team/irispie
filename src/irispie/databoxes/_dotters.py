"""
"""

#[
from __future__ import annotations

from typing import (Self, )

from . import main as _databoxes

import documark as _dm
#]


class Dotter:
    r"""
................................................................................
==Dotter Class==

The `Dotter` class facilitates conversion between an object and a databox-like
structure. It offers methods to initialize the class instance from a databox 
and to update a databox with the instance's attributes.

This class serves as a utility for managing data, ensuring seamless attribute
synchronization with external data containers. The structure adheres to a simple
dictionary-based update mechanism.

................................................................................
    """

    #[

    @classmethod
    @_dm.reference(category="TBD", )
    def from_databox(
        klass,
        db: _databoxes.Databox,
    ) -> Self:
        r"""
................................................................................
==Create a Dotter instance from a Databox==

This method initializes a `Dotter` instance by updating its attributes
with the key-value pairs from the provided databox.

It facilitates smooth initialization from external data containers,
leveraging the dynamic update capability of Python's `__dict__`.

### Input arguments ###
???+ input
    `klass`: The class to instantiate.
    - Typically the `Dotter` class itself.

???+ input
    `db`: An instance of `_databoxes.Databox`.
    - Contains key-value pairs to populate the `Dotter` instance.

### Returns ###
???+ returns
    `Self`: An initialized `Dotter` instance.
    - Contains attributes mirrored from the provided databox.

### Example ###
```python
    db = Databox(key1="value1", key2="value2")
    dotter_instance = Dotter.from_databox(db)
```
................................................................................
        """
        self = klass()
        self.__dict__.update(db, )
        return self

    @_dm.reference(category="TBD", )
    def to_databox(
        self,
        db: _databoxes.Databox,
    ) -> None:
        r"""
................................................................................
==Update a Databox with Dotter attributes==

This method synchronizes the provided databox with the attributes
of the `Dotter` instance, effectively exporting the current state
to the databox.

### Input arguments ###
???+ input
    `db`: An instance of `_databoxes.Databox`.
    - Will be updated with the `Dotter` instance's attributes.

### Returns ###
???+ returns
    `None`: The databox is updated in-place.

### Example ###
```python
    dotter_instance.to_databox(db)
```
................................................................................
        """
        db.update(self.__dict__, )

    #]


class DotterMixin:
    r"""
................................................................................
==DotterMixin Class==

The `DotterMixin` class enables seamless integration of `Dotter` functionality
into other classes. It provides context manager support for managing and
synchronizing `Dotter` instances in a structured manner.

The mixin ensures that a `Dotter` instance is appended upon entering a context
and removed after the context completes, updating the source accordingly.

................................................................................
    """

    #[

    def __enter__(
        self,
    ) -> Self:
        r"""
................................................................................
==Enter a Dotter-managed context==

This method appends a new `Dotter` instance to the `_dotters` stack
of the current instance. It facilitates context-based management
of synchronized `Dotter` instances.

### Input arguments ###
???+ input
    None

### Returns ###
???+ returns
    `Self`: A newly created `Dotter` instance.

### Example ###
```python
    with dotter_mixin_instance as dotter:
        # Perform operations with `dotter`.
```
................................................................................
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
        r"""
................................................................................
==Exit a Dotter-managed context==

This method removes the last `Dotter` instance from the `_dotters` stack
and updates the current instance with its attributes.

### Input arguments ###
???+ input
    `exc_type`: Exception type, if any.
    - Captures the class of any exception raised within the context.

???+ input
    `exc_value`: Exception value, if any.
    - Contains the actual exception object.

???+ input
    `traceback`: Traceback object, if any.
    - Provides the stack trace of the exception.

### Returns ###
???+ returns
    `None`: Attributes of the `Dotter` instance are updated in-place.

### Example ###
```python
    with dotter_mixin_instance:
        # Perform operations.
```
................................................................................
        """
        current_dotter = self._dotters.pop()
        current_dotter.to_databox(self, )

    #]
