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


def make_lambda(
    args: Iterable[str],
    expression: str,
    globals: dict[str, Any] | None = None,
) -> tuple[Callable, str, dict[str, Any]]:
    """
    Create lambda from the list of args and an expression
    """
    #[
    func_str = f"lambda {', '.join(args, )}: {str(expression)}"
    func = eval(func_str, globals, )
    return func, func_str, globals
    #]


def prepare_globals(
    *,
    custom_functions: dict[str, Any] | None = None,
    builtins: dict[str, Callable] | None = None,
) -> dict[str, Any]:
    """
    Create globals dict restricting builtins by default
    """
    #[
    globals_ = _cp.deepcopy(custom_functions, ) if custom_functions else {}
    globals_["__builtins__"] = builtins
    globals_ = _adaptations.add_function_adaptations_to_custom_functions(globals_, )
    return globals_
    #]

