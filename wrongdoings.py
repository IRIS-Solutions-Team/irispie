"""Handle wrongdoings"""


#[
from __future__ import annotations

import warnings as wa_
from typing import (TypeAlias, Literal, NoReturn, )
from collections.abc import (Iterable, )
#]


_How: TypeAlias = Literal["error"] | Literal["warning"] | Literal["silent"]

_PREFIX = "==> "


class IrisPieError(Exception):
    def __init__(self, description):
        message = prepare_message(description)
        super().__init__(message)


class IrisPieWarning(UserWarning):
    pass


def throw(
    how: _How,
    message: str | Iterable[str],
    **args,
) -> NoReturn:
    #[
    match how:
        case "error":
            raise IrisPieError(message)
        case "warning":
            message = _prepare_message(message)
            message = "\n\nIrisPieWarning: " + message + "\n"
            wa_.warn(message, IrisPieWarning, stacklevel=2, )
        case "silent":
            pass
    #]


def prepare_message(message):
    #[
    if isinstance(message, str):
        message = _PREFIX + message
    else:
        message = ("\n"+_PREFIX).join(message)
    return message
    #]

