
from irispie import *

source_string = r"""
    !parameters
    !for <sectors> !do 
    ?
    !end
    ;
    !transition-variables
    x
    !transition-equations
    x = 0;
"""

context = {
    "sectors": ["a","b","c"],
}

m = Simultaneous.from_string(
    source_string,
    context=context,
)

