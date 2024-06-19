

class A:
    x = [1, 2, 3, 4, 5]
    print(x)
    for i in x:
        print(i)
    b = x[2]
    print(dir())

class B(A):
    print(dir())
    z = 100
    def __init__(self):
        pass

i = B()
j = B()

