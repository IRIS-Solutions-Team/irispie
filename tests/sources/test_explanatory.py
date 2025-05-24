

import irispie as ir
import numpy as _np
import scipy as _sp
import sys
from irispie.incidences import permutations as _permutations

source_code = """

!equations

    "aaa"
    a = 0.8*a[-1];
    y = b{-1};
    roc(b) = 0.5*a + c[-2] + d[+2];
    s === a + b + c + y;
    u = u[-1];

"""

z = ir.Sequential.from_string(source_code, )

sys.exit()
qid_to_name = z.create_qid_to_name()

d = ir.Databank()
im = z.incidence_matrix
lhs_names = z.lhs_names

# #ims = _sp.sparse.csc_matrix(im)
# ims = im
# lu = _sp.sparse.linalg.splu(ims, permc_spec="MMD_ATA")
# im1 = im;
# im1 = im1[lu.perm_r, :]
# im1 = im1[:, lu.perm_c]
# print(im1)

s = z
s.is_sequential
rows = s.sequentialize()


# (rows, columns), _, info = _permutations.sequentialize(im)
# im = im[rows, :]
# im = im[:, rows]
# print(im)
# 

start_sim = ir.yy(2020)
end_sim = ir.yy(2024)
sim_rng = ir.yy(2020) >> ir.yy(2024)
db_rng = start_sim-10 >> end_sim+2
db = ir.Databank.for_simulation(s, db_rng, func=_np.random.uniform)
for n in s.res_names:
    db[n][...] = 0

sim = s.simulate(db, sim_rng, )


