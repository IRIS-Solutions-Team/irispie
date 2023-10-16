"""
Handle exceptions and warnings
"""


#[
from __future__ import annotations

from typing import (TypeAlias, Literal, Callable, NoReturn, )
from collections.abc import (Iterable, )
import warnings as _wa
import os as _os
#]

_WARN_SKIPS = (_os.path.dirname(__file__), )
HOW: TypeAlias = Literal["error", "warning", "silent"]


_PLAIN_PREFIX = ""
_LIST_PREFIX = "»» "


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


def _raise(
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


def _raise_as_error(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    raise IrisPieError(message)
    #]


def _raise_as_warning(
    message: str | Iterable[str],
) -> None:
    """
    """
    #[
    message = _prepare_message(message)
    message = "\nIrisPieWarning: " + message
    try:
        _wa.warn(message, IrisPieWarning, skip_file_prefixes=_WARN_SKIPS, )
    except TypeError:
        _wa.warn(message, IrisPieWarning, )
    #]


def _raise_as_silent(
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
        _raise_as_warning(message)
        return func(*args, **kwargs)
    return wrapper
    #]


_RESOLVE_HOW = {
    "error": _raise_as_error,
    "warning": _raise_as_warning,
    "silent": _raise_as_silent,
}


class _Stream:
    """
    """
    #[

    def __init__(
        self,
        title: str,
        /,
    ) -> None:
        """
        """
        self.title = (title, )
        self.messages = ()

    def add(
        self,
        message: str,
    ) -> None:
        ...

    def _raise(self, /, ) -> None:
        ...

    @property
    def final_message(self, /, ) -> tuple[str, ...]:
        return self.title + self.messages

    #]


class ErrorStream(_Stream):
    """
    """
    #[

    def add(
        self,
        message: str,
    ) -> NoReturn:
        """
        """
        self.messages += (message, )
        raise IrisPieError(self.final_message, )

    #]


class WarningStream(_Stream):
    """
    """
    #[

    def add(
        self,
        message: str,
    ) -> None:
        self.messages += (message, )

    def _raise(self, /, ) -> None:
        """
        """
        if self.messages:
            _raise_as_warning(self.final_message)

    #]


class SilentStream(_Stream):
    """
    """
    #[

    def add(self, *args, **kwargs, ) -> None:
        pass

    def _raise(self, *args, **kwargs, ) -> None:
        pass

    #]


STREAM_FACTORY = {
    "error": ErrorStream,
    "warning": WarningStream,
    "silent": SilentStream,
}


