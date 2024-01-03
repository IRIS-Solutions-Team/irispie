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
    context: dict[str, Any] | None,
    /,
) -> tuple[Callable, str, dict[str, Any]]:
    """
    Create lambda from the list of args and an expression
    """
    #[
    globals_ = _prepare_globals(context, )
    func_str = f"lambda {', '.join(args, )}: {str(expression)}"
    func = eval(func_str, globals_, )
    return func, func_str, globals_
    #]


def _prepare_globals(
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    """
    #[
    globals_ = _cp.deepcopy(context, ) if context else {}
    globals_["__builtins__"] = None
    globals_ = _adaptations.add_function_adaptations_to_context(globals_, )
    return globals_
    #]

