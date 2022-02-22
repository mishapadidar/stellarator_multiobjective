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
n_points_per = 50

# make the discretization
#max_pert = 0.1
#ub = max_pert*np.ones(2)
#lb = -max_pert*np.ones(2)
ub = np.array([0.075,0.075])
lb = np.array([-0.05,-.1])
T1 = np.linspace(lb[0],ub[0],n_points_per)
T2 = np.linspace(lb[1],ub[1],n_points_per)
# make a grid
X1,X2 = np.meshgrid(T1,T2)

# get the directions
e1 = np.eye(dim_x)[idx1]
e2 = np.eye(dim_x)[idx2]

# make a list of points
X = np.zeros((0,dim_x))
for ii in range(n_points_per):
  for jj in range(n_points_per):
    pp = np.zeros_like(x0)
    pp[idx1] = X1[ii,jj]
    pp[idx2] = X2[ii,jj]
    pp += x0
    X = np.append(X,pp.reshape((1,-1)),axis=0)
# evaluate the points
FX = prob.evalp(X)
# reshape
FX = FX.T.reshape((dim_F,n_points_per,n_points_per))

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
