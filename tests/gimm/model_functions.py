import numpy as np_


def fun1(x):
    return np_.log(x)

def fun2(x):
    return np_.log(x) + fun1(2*x)


