"""
Dynamic function makers
"""


#[
from __future__ import annotations

from collections.abc import (Iterable, )
from typing import (Any, Callable, )
import copy as _cp

from .aldi import adaptations as _adaptations
#]


def make_function(
    func_name: str,
    args: Iterable[str],
    expression: str,
    context: dict[str, Any] | None,
    /,
) -> tuple[Callable, str, dict[str, Any]]:
    """
    Create lambda from the list of args and an expression
    """
    #[
    globals_ = _prepare_globals(context, )
    args_string = ', '.join(args, )
    func_str = f"def {func_name}({args_string}): return {str(expression)}"
    exec(func_str, globals_, )
    return globals_[func_name], func_str, globals_,
    #]


def _prepare_globals(
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    """
    #[
    globals_ = (context or {}) | {"__builtins__": {}}
    globals_ = _adaptations.add_function_adaptations_to_context(globals_, )
    return globals_
    #]

