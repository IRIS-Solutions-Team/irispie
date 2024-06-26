"""
"""


#[
from __future__ import annotations

import dill as _di
import json as _js

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, )
#]


__all__ = (
    "save",
    "load",
    "save_json",
    "load_json",
)


_DEFAULT_JSON_SETTINGS = {
    "indent": 4,
}


def save(filename: str, object_to_save: Any, ) -> None:
    r"""
................................................................................

................................................................................
    """
    with open(filename, "wb", ) as fid:
        _di.dump(object_to_save, fid, )


def load(filename: str, ) -> Any:
    r"""
................................................................................

................................................................................
    """
    with open(filename, "rb", ) as fid:
        return _di.load(fid, )


def save_json(
    filename: str,
    object_to_save: Any,
    **kwargs,
) -> None:
    r"""
................................................................................


................................................................................
    """
    json_settings = _DEFAULT_JSON_SETTINGS | kwargs
    with open(filename, "wt", ) as fid:
        _js.dump(object_to_save, fid, **json_settings, )


def load_json(filename: str, **kwargs, ) -> Any:
    """
    """
    with open(filename, "rt", ) as fid:
        return _js.load(fid, )


def save_text(
    filename: str,
    text: str,
) -> None:
    """
    """
    with open(filename, "wt", ) as fid:
        fid.write(text, )


def load_text(
    filename: str,
) -> str:
    """
    """
    with open(filename, "rt", ) as fid:
        return fid.read()

