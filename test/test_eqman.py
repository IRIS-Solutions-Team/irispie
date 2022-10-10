

# import sys, importlib  as il
# sys.path.append('..')
# 
# import modiphy.equations as eq
# il.reload(eq)

eqs = list()

for _ in range(1000):
    eqs.extend([
        eq.Equation(1, "a + b{-1} = c{+2} - 0.5*a!!a + b"),
        eq.Equation(2, "z - b{+1} = 2*a"),
    ])


names = eq.names_from_equations(eqs)

name_to_id = { name: id for id, name in enumerate(names) }

xs, inc, *_ = eq.xtrings_from_equations(eqs, name_to_id)


xsd =  [ i.split("!!")[-1] for i in xs ]
xss =  [ i.split("!!")[0] for i in xs ]

