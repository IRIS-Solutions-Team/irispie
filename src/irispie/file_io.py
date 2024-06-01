"""
"""

#[
from __future__ import annotations

import dill as _di
import json as _js
#]


__all__ = (
    "save",
    "load",
    "save_json",
    "load_json",
)


def save(filename, obj, ):
    with open(filename, "wb", ) as fid:
        _di.dump(obj, fid, )


def load(filename, ):
    with open(filename, "rb", ) as fid:
        return _di.load(fid, )


def save_json(filename: str, something, *, indent: int = 4, **kwargs, ):
    """
    """
    with open(filename, "wt", ) as fid:
        _js.dump(something, fid, indent=4, **kwargs, )


def load_json(filename: str, **kwargs, ):
    """
    """
    with open(filename, "rt", ) as fid:
        return _js.load(fid, )


