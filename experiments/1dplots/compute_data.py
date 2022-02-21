import numpy as np
import sys
sys.path.append("../../problem")
sys.path.append("../../utils")
import qh_prob1
import pickle


# load the problem
vmec_input = "../../problem/input.nfp4_QH_warm_start"
prob = qh_prob1.QHProb1(vmec_input)
x0 = prob.x0
dim_x = prob.dim_x
dim_F = prob.dim_F
np.random.seed(0) # match the seeds

# discretization parameters
n_directions = dim_x
n_points_per = 400 # total points per direction

# make the discretization
max_pert = 0.5
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -9,-3
n2 = int((n_points_per - n1)/2)
T2 = np.logspace(min_log,max_log,n2)
T2 = np.hstack((-T2,T2))
T = np.sort(np.unique(np.hstack((T1,T2))))


# use an orthogonal frame
Q = np.eye(dim_x)

# storage
X = np.zeros((n_directions,n_points_per,dim_x))
FX = np.zeros((n_directions,n_points_per,dim_F))

for ii in range(n_directions):
    print(f"direction {ii}/{dim_x}")
    sys.stdout.flush()
    # eval point
    Y = x0 + Q[ii]*np.reshape(T,(-1,1))
    fY = prob.evalp(Y)
    # save it
    X[ii] = np.copy(Y)
    FX[ii] = np.copy(fY)

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
