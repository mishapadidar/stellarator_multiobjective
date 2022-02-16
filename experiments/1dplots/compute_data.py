import numpy as np
import sys
sys.path.append("../../problem")
import qh_prob1
import pickle

# load the problem
vmec_input = "../../problem/input.nfp4_QH_warm_start"
n_partitions = 1
prob = qh_prob1.QHProb1(vmec_input,n_partitions)
x0 = prob.x0
dim_x = prob.dim_x
dim_F = prob.dim_F

# discretization parameters
n_directions = dim_x
n_points_per = 10

# make the discretization
max_pert = 0.1
ub = max_pert
lb = -max_pert
T = np.linspace(lb,ub,n_points_per)

# use an orthogonal frame
Q = np.eye(dim_x)

# storage
X = np.zeros((n_directions,n_points_per,dim_x))
FX = np.zeros((n_directions,n_points_per,dim_F))

for ii in range(n_directions):
  for jj,tt in enumerate(T):
    # eval point
    xx = np.copy(x0 + tt*Q[ii])
    fx = prob.eval(xx)
    # save it
    X[ii,jj] = np.copy(xx)
    FX[ii,jj] = np.copy(fx)

# dump a pickle file
outfile = "1dplot_data.pickle"
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['n_directions'] = n_directions
outdata['n_points_per'] = n_points_per
outdata['Q'] = Q
outdata['T'] = T
pickle.dump(outdata,open(outfile,"wb"))
