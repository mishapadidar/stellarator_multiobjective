import numpy as np
import sys
sys.path.append("../../problem")
import qh_prob1
import pickle

# load the problem
vmec_input = "../../problem/input.nfp4_QH_warm_start"
n_partitions = 2
prob = qh_prob1.QHProb1(vmec_input,n_partitions)
x0 = prob.x0
dim_x = prob.dim_x
dim_F = prob.dim_F

# discretization parameters
n_directions = dim_x
n_points_per = 200 # total points per direction

# make the discretization
max_pert = 0.5
ub = max_pert
lb = -max_pert
T1 = np.linspace(lb,ub,int(n_points_per/2))
min_log,max_log = -4,-1
T2 = np.logspace(min_log,max_log,int(n_points_per/4))
T2 = np.hstack((-T2,T2))
T = np.sort(np.unique(np.hstack((T1,T2))))


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
