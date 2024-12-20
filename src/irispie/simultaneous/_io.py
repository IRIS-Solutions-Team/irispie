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
    r"""
    ................................................................................
    ==Class: Inlay==

    Handles serialization and deserialization operations, providing methods to 
    save and load objects using various formats including Pickle, Dill, and JSON. 
    The class is designed to ensure seamless input-output functionality with 
    portable and standard file types.

    This class is essential for scenarios requiring persistent storage or transfer 
    of Python objects between systems or processes.

    Attributes:
        - None (Attributes are dynamically created and managed during operations)
    ................................................................................
    """
    #[

    def to_pickle_bytes(self, **kwargs, ) -> bytes:
        r"""
        ................................................................................
        ==Method: to_pickle_bytes==

        Serializes the object into a byte stream using the Pickle module. This method 
        is suitable for in-memory storage or transfer of serialized objects.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Pickle serialization process.

        ### Returns ###
        ???+ returns "bytes"
            A byte stream representing the serialized object.

        ### Example ###
        ```python
            data = obj.to_pickle_bytes()
        ```
        ................................................................................
        """
        return _pk.dumps(self, **kwargs, )

    def to_pickle_file(self, file_name: str, **kwargs, ) -> None:
        r"""
        ................................................................................
        ==Method: to_pickle_file==

        Serializes the object and saves it to a file using the Pickle module. The 
        method writes the object in binary format to ensure compatibility.

        ### Input arguments ###
        ???+ input "file_name: str"
            The name of the file to save the serialized object.
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Pickle serialization process.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.to_pickle_file("data.pkl")
        ```
        ................................................................................
        """
        with open(file_name, "wb", ) as file:
            _pk.dump(self, file, **kwargs, )

    def to_dill_bytes(self, **kwargs, ) -> bytes:
        r"""
        ................................................................................
        ==Method: to_dill_bytes==

        Serializes the object into a byte stream using the Dill module. This method 
        extends Pickle's functionality, supporting more complex Python objects.

        ### Input arguments ###
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Dill serialization process.

        ### Returns ###
        ???+ returns "bytes"
            A byte stream representing the serialized object.

        ### Example ###
        ```python
            data = obj.to_dill_bytes()
        ```
        ................................................................................
        """
        return _dl.dumps(self, **kwargs, )

    def to_dill_file(self, file_name: str, **kwargs, ) -> None:
        r"""
        ................................................................................
        ==Method: to_dill_file==

        Serializes the object and saves it to a file using the Dill module. This 
        method is ideal for saving objects that Pickle may not support.

        ### Input arguments ###
        ???+ input "file_name: str"
            The name of the file to save the serialized object.
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Dill serialization process.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.to_dill_file("data.dill")
        ```
        ................................................................................
        """
        with open(file_name, "wb", ) as file:
            _dl.dump(self, file, **kwargs, )

    @classmethod
    def from_pickle_file(klass, file_name: str, **kwargs, ) -> Self:
        r"""
        ................................................................................
        ==Method: from_pickle_file==

        Deserializes an object from a Pickle file. This method reads a binary Pickle 
        file and reconstructs the original object.

        ### Input arguments ###
        ???+ input "file_name: str"
            The name of the Pickle file containing the serialized object.
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Pickle deserialization process.

        ### Returns ###
        ???+ returns "Self"
            The deserialized object instance.

        ### Example ###
        ```python
            obj = Inlay.from_pickle_file("data.pkl")
        ```
        ................................................................................
        """
        with open(file_name, "rb", ) as file:
            return _pk.load(file, **kwargs, )

    @classmethod
    def from_dill_file(klass, file_name: str, **kwargs, ) -> Self:
        r"""
        ................................................................................
        ==Method: from_dill_file==

        Deserializes an object from a Dill file. This method reads a binary Dill file 
        and reconstructs the original object, including objects not supported by Pickle.

        ### Input arguments ###
        ???+ input "file_name: str"
            The name of the Dill file containing the serialized object.
        ???+ input "**kwargs"
            Additional keyword arguments to customize the Dill deserialization process.

        ### Returns ###
        ???+ returns "Self"
            The deserialized object instance.

        ### Example ###
        ```python
            obj = Inlay.from_dill_file("data.dill")
        ```
        ................................................................................
        """
        with open(file_name, "rb", ) as file:
            return _dl.load(file, **kwargs, )

    def to_portable_file(self, file_name: str, **kwargs, ) -> None:
        r"""
        ................................................................................
        ==Method: to_portable_file==

        Serializes the object into a portable JSON format and saves it to a text file. 
        This method ensures compatibility across different platforms and systems.

        ### Input arguments ###
        ???+ input "file_name: str"
            The name of the file to save the serialized object.
        ???+ input "**kwargs"
            Additional keyword arguments to customize the JSON serialization process.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.to_portable_file("data.json")
        ```
        ................................................................................
        """
        portable = self._serialize_to_portable()
        with open(file_name, "wt", ) as file:
            _js.dump(portable, file, **kwargs, )

    #]

