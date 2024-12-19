"""
Helper for functionalizing methods in time series.

This module defines a template for transforming class methods into functional 
alternatives. The main utility, `FUNC_STRING`, is used to dynamically generate 
functions that operate on copies of the input object, ensuring immutability for 
functional programming paradigms.
"""


FUNC_STRING = """
def {n}(self, *args, **kwargs, ) -> Series:
    new = self.copy()
    out = new.{n}(*args, **kwargs, )
    if isinstance(out, tuple):
        return new, *out
    elif out is not None:
        return new, out
    else:
        return new
"""

