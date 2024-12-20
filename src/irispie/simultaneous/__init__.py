"""
................................................................................
==Module for Simultaneous Models==

This `__init__.py` file serves as the initialization point for the 
`simultaneous` package. It imports the main components and consolidates 
the public API by re-exporting items from the `main` module.

### Summary of Exports ###
- The module imports all the components from `main`.
- The `__all__` variable defines the public API for the package by 
  including all elements from `main_all`.

### Key Definitions ###

#### `from .main import *`
Imports all symbols from the `main` module into the namespace of the 
`simultaneous` package.

#### `__all__`
Defines the public API for the package by including all the symbols listed in 
`main_all`.

### Example Usage ###
```python
    from simultaneous import *
    print(dir())
"""

from .main import *
from .main import __all__ as main_all

__all__ = (
    *main_all,
)

