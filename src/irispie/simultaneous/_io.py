"""
Input-output operations
"""


#[

from __future__ import annotations

import json as _js
import pickle as _pk
import dill as _dl

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self

#]


class Inlay:
    """
    """
    #[

    def to_json_string(self, **kwargs, ) -> None:
        """
        """
        serialized = self.serialize()
        return _js.dumps(serialized, **kwargs, )

    def to_json_file(self, file_name: str, **kwargs, ) -> None:
        """
        """
        serialized = self.serialize()
        with open(file_name, "wt", ) as file:
            _js.dump(serialized, file, **kwargs, )

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
    def from_json_string(klass, string: str, **kwargs, ) -> Self:
        """
        """
        serialized = _js.loads(string, **kwargs, )
        return klass.deserialize(serialized, )

    @classmethod
    def from_json_file(klass, file_name: str, **kwargs, ) -> Self:
        """
        """
        with open(file_name, "rt", ) as file:
            serialized = _js.load(file, **kwargs, )
            return klass.deserialize(serialized, )

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

    #]

