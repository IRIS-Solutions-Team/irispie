"""
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

