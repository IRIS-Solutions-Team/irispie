"""
Mixin for plan registers
"""


#[
from __future__ import annotations

from .. import wrongdoings as _wrongdoings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, NoReturn, Any, Iterable, EllipsisType
#]


class Mixin:
    """
    """
    #[

    def _initialize_registers(
        self,
        plannable,
        default_value: Callable,
    ) -> None:
        """
        """
        for n in self._registers:
            can_be_name = f"can_be_{n}"
            register = {
                n: default_value(n, )
                for n in getattr(plannable, can_be_name, )
            } if hasattr(plannable, can_be_name, ) else {}
            setattr(self, can_be_name, tuple(register.keys()))
            setattr(self, f"_{n}_register", register)

    def get_register_by_name(self, name: str, /, ) -> dict[str, Any]:
        """
        """
        full_name = f"_{name}_register"
        return getattr(self, full_name, ) if hasattr(self, full_name, ) else None

    @staticmethod
    def _resolve_register_names(
        register: dict | None,
        names: Iterable[str] | str | EllipsisType,
    ) -> tuple[str] | NoReturn:
        """
        """
        keys = tuple(register.keys()) if register else ()
        if names is Ellipsis:
            names = keys
        elif isinstance(names, str):
            names = (names, )
        else:
            names = tuple(names)
        _validate_register_names(register, names, )
        return names

    #]


def _validate_register_names(
    register: dict | None,
    names: Iterable[str],
) -> None | NoReturn:
    """
    """
    #[
    keys = tuple(register.keys()) if register else ()
    invalid = tuple(n for n in names if n not in keys)
    if invalid:
        message = (f"These names are not valid in the register:", ) + invalid
        raise _wrongdoings.IrisPieCritical(message, )
    #]

