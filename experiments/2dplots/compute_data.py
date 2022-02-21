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

# choose two direction indexes
idx1 = 6
idx2 = 15
# number of points per direction (N^2 points total)
n_points_per = 100

# make the discretization
max_pert = 0.25
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -5,-1
n2 = int((n_points_per - n1)/2)
T2 = np.logspace(min_log,max_log,n2)
T2 = np.hstack((-T2,T2))
T = np.sort(np.unique(np.hstack((T1,T2))))

# get the directions
e1 = np.eye(dim_x)[idx1]
e2 = np.eye(dim_x)[idx2]

# make a grid
X1,X2 = np.meshgrid(T,T)
X = np.zeros((0,dim_x))
# make a list of points
for ii in range(n_points_per):
  for jj in range(n_points_per):
    pp = np.copy(x0)
    pp[idx1] = X1[ii,jj]
    pp[idx2] = X2[ii,jj]
    X = np.append(X,pp.reshape((1,-1)),axis=0)
# evaluate the points
FX = prob.evalp(X)
# reshape
FX = FX.T.reshape((dim_F,n_points_per,n_points_per))
print(FX)

# dump a pickle file
outfile = f"2dplot_data_{idx1}_{idx2}.pickle"
outdata = {}
outdata['X1'] = X1
outdata['X2'] = X2
outdata['FX'] = FX
outdata['n_points_per'] = n_points_per
outdata['idx1'] = idx1
outdata['idx2'] = idx2
outdata['x0'] = x0
outdata['T'] = T
pickle.dump(outdata,open(outfile,"wb"))