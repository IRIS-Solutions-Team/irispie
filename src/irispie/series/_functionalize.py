"""
"""


FUNC_STRING = """
def {n}(self, *args, **kwargs, ) -> Series:
    new = self.copy()
    new.{n}(*args, **kwargs, )
    return new
"""

