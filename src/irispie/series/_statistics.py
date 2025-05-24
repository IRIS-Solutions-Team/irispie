"""
"""


#[

from __future__ import annotations

import numpy as _np
import textwrap as _tw

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from numbers import Real
    from . import Series, AxisType

#]


_NANABLE_FUNCTIONS = (
    "sum",
    "prod",
    "mean",
    "median",
    "std",
    "var",
    "max",
    "min",
    "percentile",
    "quantile",
)

_NAN_FUNCTIONS = tuple( f"nan{n}" for n in _NANABLE_FUNCTIONS )

__all__ = _NANABLE_FUNCTIONS + _NAN_FUNCTIONS


class Inlay:
    """
    """
    #[
    for n in __all__:
        code = f"""
            def {n}(
                self: Self,
                *args,
                **kwargs,
            ) -> None:
                r'''
                Function {n}
                '''
                num_periods = self.data.shape[0]
                self.data = _np.{n}(self.data, *args, axis=1, **kwargs, ).T.reshape(num_periods, -1, )
                self.trim()
        """
        exec(_tw.dedent(code, ), )
    #]



for n in __all__:
    code = f"""
        def {n}(
            self: Series,
            *args,
            axis: AxisType = 1,
            unpack_singleton: bool = True,
            **kwargs,
        ) -> Real | list[Real] | Series:
            if axis == 0:
                result = _np.{n}(self.data, *args, axis=0, **kwargs, ).T.tolist()
                if unpack_singleton and len(result) == 1:
                    result = result[0]
                return result
            elif axis == 1:
                new = self.copy()
                new.{n}(*args, **kwargs, )
                return new
            else:
                raise ValueError(f'Axis must be 0 or 1; got {{axis}}')
    """
    exec(_tw.dedent(code, ), )

