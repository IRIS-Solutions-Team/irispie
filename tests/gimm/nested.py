
import math as ma_

def create_func(x, string, context):
    x.func = eval("lambda x:  " + string, context)

