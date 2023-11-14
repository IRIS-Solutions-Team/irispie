"""
"""

#[
from __future__ import annotations

from typing import (Protocol, )
#]


class FromFileableProtocol(Protocol, ):
    """
    """
    #[

    def from_string(source_string: str, /, **kwargs, ) -> Self: ...

    #]


class FromFileMixin:
    """
    """
    #[

    @classmethod
    def from_file(
        klass,
        source_files: str | Iterable[str],
        /,
        **kwargs,
    ) -> Self:
        """
        Create a new object from source file(s)
        """
        source_string = _combine_source_files(source_files, )
        return klass.from_string(source_string, **kwargs, )

    #]


def _combine_source_files(
    source_files: str | Iterable[str],
    /,
    joint: str = "\n\n",
) -> str:
    """
    """
    #[
    if isinstance(source_files, str):
        source_files = [source_files]
    source_strings = []
    for f in source_files:
        with open(f, "r") as fid:
            source_strings.append(fid.read(), )
    return "\n\n".join(source_strings, )
    #]

