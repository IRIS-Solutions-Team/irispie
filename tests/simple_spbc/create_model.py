
import sys
sys.path.append('../..')
from irispie import *
from irispie.parsers import preparser as pp
import scipy as sp_

m = Model.from_file("spbc.model", flat=True, )

m.assign(
    alpha = 1,
    beta = 0.985**(1/4),
    beta0 = 0,
    beta1 = 0.8,
    gamma = 0.60,
    delta = 0.03,
    pi = 1,
    eta = 6,
    k = 10,
    psi = 0.25,

    chi = 0.85,
    xiw = 60,
    xip = 300,
    rhoa = 0.90,
    rhoterm20 = 0.80,

    rhor = 0.85,
    kappap = 4,
    kappan = 0,

    Short_ = 0,
    Long_ = 0,
    Infl_ = 0,
    Growth_ = 0,
    Wage_ = 0,

    A = 1,
    P = 1,
    Pk = 5,
)


select_index = lambda q: [ 
    i for i, j in enumerate(q) 
    if j.human not in ["A", "P", "Short", "Infl", "Long", "Growth", "Wage"]
]

q1 = m._get_quantities(kind=TRANSITION_VARIABLE | MEASUREMENT_VARIABLE)
s1 = m.create_steady_evaluator(quantities=q1)
ix1 = select_index(q1)

q2 = m._get_quantities(kind=TRANSITION_VARIABLE | MEASUREMENT_VARIABLE)
q2 = filter_quantities_by_name(q2, exclude_names=["A", "P"])
s2 = m.create_steady_evaluator(quantities=q2, print_iter=True)
ix2 = select_index(q2)

r1 = sp_.optimize.root(
    s1.eval_with_jacobian,
    s1.initial_guess,
    method="lm",
    jac=True,
)

r2 = sp_.optimize.root(
    s2.eval_with_jacobian,
    s2.initial_guess,
    method="lm",
    jac=True,
)

s3 = m.create_steady_evaluator(quantities=q2, print_iter=False, flat=False)
s3.update()
j3 = s3._jacobian.eval(s3._x, None)

