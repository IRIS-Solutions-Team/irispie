"""
Mixin for plan registers
"""


#[
from __future__ import annotations

from typing import (TYPE_CHECKING, )

from .. import wrongdoings as _wrongdoings

if TYPE_CHECKING:
    from typing import (Callable, )
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

    def _get_register_by_name(self, name: str, /, ) -> dict[str, Any]:
        """
        """
        full_name = f"_{name}_register"
        return getattr(self, full_name, ) if hasattr(self, full_name, ) else None

    @staticmethod
    def _resolve_validate_register_names(
        register: dict | None,
        names: Iterable[str] | str | EllipsisType,
        register_name: str,
        /,
    ) -> tuple[str]:
        """
        """
        keys = tuple(register.keys()) if register else ()
        if names is Ellipsis:
            return keys
        names = tuple(names) if not isinstance(names, str) else (names, )
        invalid = tuple(n for n in names if n not in keys)
        if invalid:
            message = (f"These names cannot be {register_name}:", ) + invalid
            raise _wrongdoings.IrisPieCritical(message, )
        return names

    #]

