
import pickle as pk
# import model_functions as mf

with open("model.pkl", "rb") as f:
    m = pk.load(f)

print(m.eval(10, 20))

