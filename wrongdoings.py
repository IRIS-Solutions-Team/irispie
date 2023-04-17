"""Handle wrongdoings"""


#[
from __future__ import annotations

import warnings as wa_
from typing import (TypeAlias, Literal, NoReturn, )
#]


_How: TypeAlias = Literal["error"] | Literal["warning"] | Literal["silent"]

_PREFIX = "==> "


class IrisPieError(Exception):
    pass


class IrisPieWarning(UserWarning):
    pass


def throw(
    how: _How,
    message: str,
    **args,
) -> NoReturn:
    #[
    message = _prepare_message(message)
    match how:
        case "error":
            raise IrisPieError(message)
        case "warning":
            message = "\n\nIrisPieWarning: " + message + "\n"
            wa_.warn(message, IrisPieWarning, stacklevel=2, )
        case "silent":
            pass
    #]


def _prepare_message(message):
    #[
    if isinstance(message, str):
        message = _PREFIX + message
    else:
        message = ("\n"+_PREFIX).join(message)
    return message
    #]

