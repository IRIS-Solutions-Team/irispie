"""
Handle exceptions and warnings
"""


#[
from __future__ import annotations

from typing import (TypeAlias, Literal, Callable, )
from collections.abc import (Iterable, )
import warnings as _wa
#]


HOW: TypeAlias = Literal["error", "warning", "silent"]


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
    message = "\nIrisPieWarning: " + message
    _wa.warn(message, IrisPieWarning, stacklevel=4, )
    #]


def _throw_as_silent(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    pass
    #]


def obsolete(func):
    """
    """
    #[
    def wrapper(*args, **kwargs, ):
        message = (
            f"Function {func.__name__} is obsolete and will be removed in a future version of IrisPie. "
        )
        _throw_as_warning(message)
        return func(*args, **kwargs)
    return wrapper
    #]


_RESOLVE_HOW = {
    "error": _throw_as_error,
    "warning": _throw_as_warning,
    "silent": _throw_as_silent,
}

