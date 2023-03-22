
#[
from __future__ import annotations
from IPython import embed

import json as js_
import copy as co_
import types as ty_
import numpy as np_
import re as re_
from typing import (Self, TypeAlias, NoReturn, Literal, )
from collections.abc import (Iterable, Callable, )
from numbers import (Number, )

from . import dates as da_
#]


SourceNames: TypeAlias = Iterable[str] | str | Callable[[str], bool] | None
TargetNames: TypeAlias = Iterable[str] | str | Callable[[str], str] | None


_REPR_MAX_LEN = 70
_REPR_CONT = "..."
_REPR_INDENT = "    "
_REPR_SEPARATOR = ": "


DATABANK_OUTPUT_FORMAT_RESOLUTION = {
    "dict": lambda x: vars(x),
    "Databank": lambda x: x,
    "databank": lambda x: x,
    "json": lambda x: js_.dumps(vars(x)),
    "json4": lambda x: js_.dumps(vars(x), indent=4),
}


class EmptyDateRange(Exception):
    def __init__(self):
        super().__init__("Empty date range is not allowed in this context.")


class Databank(ty_.SimpleNamespace):
    """
    """
    #[
    @classmethod
    def _from_dict(
        cls,
        _dict: dict,
        /,
    ) -> Self:
        """
        """
        self = cls()
        for k, v in _dict.items():
            self.__setattr__(k, v)
        return self

    @classmethod
    def _for_model(
        cls,
        model,
        base_range: Iterable[da_.Dater],
        /,
        interpret_range: Literal["base"]|Literal["all"] = "base",
        deviation: bool = False,
    ) -> Self:
        range_list = [t for t in base_range]
        if not range_list:
            raise EmptyDateRange()
        start_date = t[0]
        end_date = t[-1]

    def _get_names(self: Self) -> Iterable[str]:
        """
        Get all names stored in a databank save for private attributes
        """
        return [ n for n in dir(self) if not n.startswith("_") ]

    def _to_dict(self: Self) -> dict:
        """
        Convert Databank to dict
        """
        return vars(self)

    def _copy(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> Self:
        """
        """
        new_databank = co_.deepcopy(self)
        new_databank = new_databank._rename(source_names, target_names)
        new_databank._keep(target_names)
        return new_databank

    def _rename(
        self: Self,
        /,
        source_names: SourceNames = None,
        target_names: TargetNames = None,
    ) -> Self:
        """
        """
        context_names = self._get_names()
        source_names, target_names = _resolve_source_target_names(source_names, target_names, context_names)
        for old_name, new_name in (
            z for z in zip(source_names, target_names) if z[0] in context_names and z[0]!=z[1]
        ):
            self.__dict__[new_name] = self.__dict__.pop(old_name)
        return self

    def _remove(
        self: Self,
        /,
        remove_names: SourceNames = None,
    ) -> Self:
        """
        """
        if remove_names is None:
            return self
        context_names = self._get_names()
        remove_names, *_ = _resolve_source_target_names(remove_names, None, context_names)
        for n in set(remove_names).intersection(context_names):
            del self.__dict__[n]
        return self

    def _keep(
        self: Self,
        /,
        keep_names: SourceNames = None,
    ) -> Self:
        """
        """
        if keep_names is None:
            return self
        context_names = self._get_names()
        keep_names, *_ = _resolve_source_target_names(keep_names, None, context_names)
        remove_names = set(context_names).difference(keep_names)
        return self._remove(remove_names)

    def _to_json(self, **kwargs):
        return js_.dumps(vars(self), **kwargs)

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value) -> NoReturn:
        self.__dict__[name] = value

    def __or__(self, other) -> Self:
        new = co_.deepcopy(self)
        new.__dict__.update(other.__dict__)
        return new

    def __repr__(self, /, ) -> NoReturn:
        max_len = max(len(str(k)) for k in self.__dict__.keys())
        s = [ _REPR_INDENT + str(k).rjust(max_len) + _REPR_SEPARATOR + _databank_repr(v) for k, v, in self.__dict__.items() ]
        return "\n".join(s)

    def __str__(self, /, ) -> NoReturn:
        return repr(self)
    #]


def _resolve_source_target_names(
    source_names: SourceNames,
    target_names: TargetNames,
    context_names: Iterable[str],
    /,
) -> tuple[Iterable[str], Iterable[str]]:
    if source_names is None:
        source_names = context_names
    if isinstance(source_names, str):
        source_names = [source_names]
    if callable(source_names):
        source_names = (n for n in existing_names if source_names(n))
    if target_names is None:
        target_names = source_names
    if isinstance(target_names, str):
        target_names = [target_names]
    if callable(target_names):
        target_names = (target_names(n) for n in source_names)
    return source_names, target_names


def _databank_repr(x, /, ) -> str:
    if x is None:
        s = "None"
    elif x is ...:
        s = "..."
    elif isinstance(x, Number) or isinstance(x, da_.Dater):
        s = str(x)
    elif isinstance(x, str):
        s = f'"{x}"'
    elif isinstance(x, np_.ndarray) or isinstance(x, list):
        s = re_.sub("\n + ", " ", repr(x))
    else:
        s = repr(type(x))
    return s if len(s)<_REPR_MAX_LEN else s[0:_REPR_MAX_LEN] + _REPR_CONT

