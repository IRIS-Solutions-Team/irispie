"""
"""


#[

from __future__ import annotations

import dill as _dl
import pickle as _pk
import json as _js

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any

#]


__all__ = (
    "save",
    "load",
    "save_dill",
    "load_dill",
    "save_pickle",
    "load_pickle",
    "save_json",
    "load_json",
    "save_text",
    "load_text",
)


def save(file_name: str, object_to_save: Any, ) -> None:
    r"""
................................................................................

................................................................................
    """
    with open(file_name, "wb", ) as fid:
        _dl.dump(object_to_save, fid, )


def load(file_name: str, ) -> Any:
    r"""
................................................................................

................................................................................
    """
    with open(file_name, "rb", ) as fid:
        return _dl.load(fid, )


def save_pickle(
    object_to_save: Any,
    file_name: str,
    **kwargs,
) -> None:
    r"""
................................................................................


................................................................................
    """
    with open(file_name, "wb", ) as fid:
        _pk.dump(object_to_save, fid, **kwargs, )


def load_pickle(
    file_name: str,
    **kwargs,
) -> Any:
    """
    """
    with open(file_name, "rb", ) as fid:
        return _pk.load(fid, **kwargs, )


def save_dill(
    object_to_save: Any,
    file_name: str,
    **kwargs,
) -> None:
    r"""
................................................................................


................................................................................
    """
    with open(file_name, "wb", ) as fid:
        _dl.dump(object_to_save, fid, **kwargs, )


def load_dill(
    file_name: str,
    **kwargs,
) -> Any:
    """
    """
    with open(file_name, "rb", ) as fid:
        return _dl.load(fid, **kwargs, )


def save_json(
    object_to_save: Any,
    file_name: str,
    **kwargs,
) -> None:
    r"""
................................................................................


................................................................................
    """
    with open(file_name, "wt", ) as fid:
        _js.dump(object_to_save, fid, **kwargs, )


def load_json(
    file_name: str,
    **kwargs,
) -> Any:
    """
    """
    with open(file_name, "rt", ) as fid:
        return _js.load(fid, **kwargs, )


def save_text(
    text: str,
    file_name: str,
    **kwargs,
) -> None:
    """
    """
    with open(file_name, "wt", ) as fid:
        fid.write(text, **kwargs, )


def load_text(
    file_name: str,
    **kwargs,
) -> str:
    """
    """
    with open(file_name, "rt", ) as fid:
        return fid.read(**kwargs, )


