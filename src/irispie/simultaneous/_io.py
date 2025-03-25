"""
Input-output operations
"""


#[

from __future__ import annotations

import json as _js
import pickle as _pk
import dill as _dl
import documark as _dm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self, Any

#]


class Inlay:
    """
    """
    #[

    def to_pickle_bytes(self, **kwargs, ) -> bytes:
        """
        """
        return _pk.dumps(self, **kwargs, )

    def to_pickle_file(self, file_name: str, **kwargs, ) -> None:
        """
        """
        with open(file_name, "wb", ) as file:
            _pk.dump(self, file, **kwargs, )

    def to_dill_bytes(self, **kwargs, ) -> bytes:
        """
        """
        return _dl.dumps(self, **kwargs, )

    def to_dill_file(self, file_name: str, **kwargs, ) -> None:
        """
        """
        with open(file_name, "wb", ) as file:
            _dl.dump(self, file, **kwargs, )

    @classmethod
    def from_pickle_file(klass, file_name: str, **kwargs, ) -> Self:
        """
        """
        with open(file_name, "rb", ) as file:
            return _pk.load(file, **kwargs, )

    @classmethod
    def from_dill_file(klass, file_name: str, **kwargs, ) -> Self:
        """
        """
        with open(file_name, "rb", ) as file:
            return _dl.load(file, **kwargs, )

    @_dm.reference(category="serialize", )
    def to_portable_file(
        self,
        file_name: str,
        json_settings: None | dict[str, Any] = None,
        **kwargs,
    ) -> None:
        r"""
................................................................................

==Write a `Simultaneous` object to a portable JSON file==

Convert the `Simultaneous` object to a portable dictionary and write it to a
JSON file. See [`Simultaneous.to_portable`](#to_portable) for details on the
conversion to a portable dictionary.


```
self.to_portable_file(
    file_name,
    json_settings=None,
)
```


### Input arguments ###

???+ input "file_name"
    Filename under which to save the JSON file.

???+ input "json_settings"
    Optional settings to pass to `json.dump`.


### Returns ###

The method returns `None`.

................................................................................
        """
        portable = self.to_portable(**kwargs, )
        with open(file_name, "wt", ) as file:
            _js.dump(portable, file, **(json_settings or {}), )

    @classmethod
    @_dm.reference(category="constructor", )
    def from_portable_file(
        klass,
        file_name: str,
        json_settings: None | dict[str, Any] = None,
        **kwargs,
    ) -> Self:
        r"""
................................................................................

==Read a `Simultaneous` object from a portable JSON file==

Read a JSON file and convert the contents to a `Simultaneous` object. See
[`Simultaneous.from_portable`](#from_portable) for details on the conversion
from a portable dictionary.

```
self = Simultaneous.from_portable_file(
    file_name,
    json_settings=None,
)
```


### Input arguments ###

???+ input "file_name"
    Filename from which to read the JSON file.

???+ input "json_settings"
    Optional settings to pass to `json.load`.


### Returns ###

???+ returns "self"
    A new `Simultaneous` object created from the contents of the JSON file.

................................................................................

        """
        with open(file_name, "rt", ) as file:
            portable = _js.load(file, **(json_settings or {}), )
        return klass.from_portable(portable, **kwargs, )

    #]

