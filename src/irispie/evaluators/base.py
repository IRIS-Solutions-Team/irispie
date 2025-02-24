"""
Evaluators
"""


#[
from __future__ import annotations

import dataclasses as _dc

from typing import (TYPE_CHECKING, )
if TYPE_CHECKING:
    from typing import (Any, Callable, )
#]


DEFAULT_MAYBELOG_INIT_GUESS = 1/9


@_dc.dataclass
class Evaluator:
    """
    """
    #[

    eval_func_jacob: Callable

    eval_func: Callable

    eval_jacob: Callable

    update: Callable

    get_init_guess: Callable

    #]

