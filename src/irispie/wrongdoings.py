"""Handle wrongdoings"""


#[
import warnings as wa_
from typing import (TypeAlias, Literal, )
from collections.abc import (Iterable, )
#]


HOW: TypeAlias = Literal["error"] | Literal["warning"] | Literal["silent"]

_PLAIN_PREFIX = ""
_LIST_PREFIX = "××× "


class IrisPieError(Exception):
    """
    """
    #[
    def __init__(self, message, ):
        message = _prepare_message(message)
        super().__init__(message)
    #]


class IrisPieWarning(UserWarning):
    """
    """
    pass


def throw(
    how: HOW,
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    _RESOLVE_HOW[how](message)
    #]


def _prepare_message(message):
    #[
    if isinstance(message, str):
        message = _PLAIN_PREFIX + message
    else:
        message = ("\n"+_LIST_PREFIX).join(message)
    return message
    #]


def _throw_as_error(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    raise IrisPieError(message)
    #]


def _throw_as_warning(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    message = _prepare_message(message)
    message = "\n\nIrisPieWarning: " + message + "\n"
    wa_.warn(message, IrisPieWarning, stacklevel=4, )
    #]


def _throw_as_silent(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    pass
    #]


_RESOLVE_HOW = {
    "error": _throw_as_error,
    "warning": _throw_as_warning,
    "silent": _throw_as_silent,
}

