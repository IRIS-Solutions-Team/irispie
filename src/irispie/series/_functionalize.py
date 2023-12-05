"""
"""


FUNC_STRING = """
def {n}(self, *args, **kwargs, ) -> _series.Series:
    new = self.copy()
    new.{n}(*args, **kwargs, )
    return new
"""

