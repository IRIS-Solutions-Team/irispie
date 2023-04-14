"""Handle wrongdoings"""


#[
from __future__ import annotations

import warnings as wa_
from typing import (TypeAlias, Literal, )
#]


_How: TypeAlias = Literal["error"] | Literal["warning"] | Literal["silent"]


class IrisPieError(Exception):
    pass


class IrisPieWarning(UserWarning):
    pass


def throw(
    how: _How,
    message: str,
    **args,
):
    match how:
        case "error":
            message = "\n" + message
            raise IrisPieError(message)
        case "warning":
            message = "\n\nIrisPieWarning:\n" + message + "\n"
            wa_.warn(message, IrisPieWarning, stacklevel=2, )
        case "silent":
            pass


