
import model as md
import pickle as pk
# import numpy as np
import model_functions as mf

# def cdf(a, b):
    # return np.log(a) + np.log(b)

def main():

    context = {
        "cdf": mf.cdf,
    }

    m = md.Model(context)
    print(m.eval(10, 20))

    with open("model.pkl", "wb") as f:
        pk.dump(m, f)

    return m

if __name__ == "__main__":
    m = main()

