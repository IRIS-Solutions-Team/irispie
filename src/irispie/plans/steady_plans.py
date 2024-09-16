"""
Meta plans for steady-state calculations
"""


#[

from __future__ import annotations

from typing import (TYPE_CHECKING, Protocol, )
from numbers import (Real, )
import textwrap as _tw
import documark as _dm

from ..conveniences import copies as _copies
from ..series.main import (Series, )
from . import _registers as _registers
from . import _pretty as _pretty

if TYPE_CHECKING:
    from typing import (Iterable, )

#]


__all__ = (
    "SteadyPlan",
)


if TYPE_CHECKING:
    _InputNames = Iterable[str] | str | EllipsisType


class SimulationPlannableProtocol(Protocol, ):
    """
    """
    #[

    steady_can_be_exogenized: Iterable[str] | None
    steady_can_be_endogenized: Iterable[str] | None
    steady_can_be_fixed_level: Iterable[str] | None
    steady_can_be_fixed_change: Iterable[str] | None

    #]



#[
@_dm.reference(
    path=("structural_models", "steady_plans.md", ),
    categories={
        "constructor": "Creating new steady plans",
        "exogenizing": "Exogenizing and endogenizing steady-state values",
        "fixing": "Fixing steady-state values",
        "information": "Getting information about steady plans",
    },
)
#]
class SteadyPlan(
    _registers.Mixin,
    _pretty.Mixin,
    _copies.Mixin,
):
    """
    """
    #[

    _TABLE_FIELDS = ("NAME", "REGISTER", "VALUE", )

    _registers = (
        "exogenized",
        "endogenized",
        "fixed_level",
        "fixed_change",
    )

    __slots__ = (
        ()
        + tuple(f"_can_be_{r}" for r in _registers)
        + tuple(f"_{r}_register" for r in _registers)
    )

    @_dm.reference(
        category="constructor",
        call_name="SteadyPlan",
    )
    def __init__(
        self,
        model,
        **kwargs,
    ) -> None:
        r"""
················································································

==Create new steady plan object==

................................................................................
        """
        plannable = model.get_steady_plannable(**kwargs, )
        def default_value(*args, **kwargs, ):
            return False
        self._initialize_registers(plannable, default_value, )

    for method_name, register_name in (
        ("exogenize", "exogenized", ),
        ("endogenize", "endogenized", ),
        ("fix_level", "fixed_level", ),
        ("fix_change", "fixed_change", ),
    ):
        exec(_tw.dedent(f"""
            def {method_name}(self, names: _InputNames, ) -> None:
                self._write_to_register("{register_name}", names, True, )

            def un{method_name}(self, names: _InputNames, ) -> None:
                self._write_to_register("{register_name}", names, False, )

            def get_{register_name}_names(self, /, ) -> tuple[str, ...]:
                return self._get_names_from_register("{register_name}", )
        """))

    # TODO: pluralize as default
    fix_levels = fix_level
    fix_changes = fix_change

    @_dm.reference(category="fixing", )
    def fix(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Fix steady-state values==

................................................................................
        """
        self.fix_level(*args, **kwargs, )
        if self._fixed_change_register:
            self.fix_change(*args, **kwargs, )

    @_dm.reference(category="fixing", )
    def unfix(self, *args, **kwargs, ) -> None:
        r"""
................................................................................

==Unfix steady-state values==

................................................................................
        """
        self.unfix_level(*args, **kwargs, )
        if self._fixed_change_register:
            self.unfix_change(*args, **kwargs, )

    def swap(self, *args, ) -> None:
        """
        """
        for a in args:
            self.exogenize(a[0], )
            self.endogenize(a[1], )

    def unswap(self, *args, ) -> None:
        """
        """
        for a in args:
            self.unexogenize(a[0], )
            self.unendogenize(a[1], )

    @property
    def is_empty(self, /, ) -> bool:
        """
        True if there are no exogenized, endogenized or fixed names in the plan
        """
        return not any( self.any_in_register(r, ) for r in self._registers )

    def any_in_register(self, register_name: str, /, ) -> bool:
        """
        """
        return any(self.get_register_by_name(register_name, ).values(), )

    def _write_to_register(
        self,
        register_name: str,
        names: _InputNames,
        new_status: bool,
    ) -> None:
        """
        """
        register = self.get_register_by_name(register_name, )
        names = self._resolve_register_names(register, names, )
        for n in names:
            register[n] = new_status

    def _get_names_from_register(
        self,
        register_name: str,
    ) -> tuple[str, ...]:
        """
        """
        register = self.get_register_by_name(register_name, )
        return tuple(
            name for name, value in register.items()
            if value
        )

    def _add_register_to_table(
        self,
        table,
        register: dict,
        action: str,
        model = None,
        *args, **kwargs,
    ) -> None:
        """
        """
        #
        missing_str = Series._missing_str
        level_values = model.get_steady_levels() if model else {}
        change_values = model.get_steady_changes() if model else {}
        parameter_values = model.get_parameters() if model else {}
        def _print_value(value: Real | None, ) -> str:
            return f"{value:g}" if value is not None else missing_str
        #
        def _get_exogenized_value(name, ):
            return (
                "("
                + _print_value(level_values.get(name, None))
                + ", "
                + _print_value(change_values.get(name, None))
                + ")"
            )
        #
        def _get_endogenized_value(name, ):
            return _print_value(parameter_values.get(name, None), )
        #
        def _get_fixed_level_value(name, ):
            return _print_value(level_values.get(name, None), )
        #
        def _get_fixed_change_value(name, ):
            return _print_value(level_values.get(name, None), )
        #
        GET_VALUE_FUNC = {
            "exogenized": _get_exogenized_value,
            "endogenized": _get_endogenized_value,
            "fixed_level": _get_fixed_level_value,
            "fixed_change": _get_fixed_change_value,
        }
        #
        def _get_value(name, action, ):
            return GET_VALUE_FUNC[action](name, )
        #
        all_rows = (
            (k, action, _get_value(k, action, ), )
            for k, v in register.items() if v
        )
        all_rows = sorted(all_rows, key=lambda row: (row[0], row[1], ), )
        for row in all_rows:
            table.add_row(row, )

    #]


