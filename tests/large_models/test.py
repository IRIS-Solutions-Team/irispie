
import irispie as _ir
import json as _js


source = """
    !variables
        !for ? = <range(1000)> !do
            x?
        !end
    !equations
        !for ?i = <range(1000)> !do
            x?i = !for ?j = <range(1,1000,70)> !do + x?j[-3] !end;
        !end
"""

m = _ir.Model.from_string(source, linear=True, )



d = {
    n: list(range(100))
    for n in ("x"+str(i) for i in range(1000))
}


