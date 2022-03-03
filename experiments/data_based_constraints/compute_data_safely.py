import numpy as np
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
from combine_bounds import combine_bounds_from_files
import safe_eval
import pickle
import glob
import time

"""
Compute the data based bound constraints with safe evaluation.
This script can only be run serially, i.e. with 
`python3 compute_data_safely.py`

Run this script after running compute_data.py because this script
cannot access parallelism.
"""

# parameters
max_iter = 100
# number of points per iteration
n_points_per = 50 # need more than 1
# growth factor ( greater than 0)
growth_factor = 1.5

# start up a safe evaluator
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
eval_script = "./qh_prob1_safe_eval.py"
dim_F = 2
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script)

# use x0 to ensure we get a feasible sample
x0_input = "../../problem/x0.nfp4_QH_warm_start_high_res.pickle"
x0 = pickle.load(open(x0_input,"rb"))


def compute_bounds(X,CX):
  """
  Find tightest bounds
  X: 2D array of points, (n,dim_x)
  CX: 2D array of 1 or zero constraint satisfaction, (n,1)
  """
  idx = np.copy(CX).flatten().astype(bool)
  lb = np.copy(np.min(X[idx],axis=0))
  ub = np.copy(np.max(X[idx],axis=0))
  return lb,ub

# initialize bounds
filelist = glob.glob("./data/samples_*.pickle")
lb0,ub0,_,__ = combine_bounds_from_files(filelist)
lb,ub = np.copy(lb0),np.copy(ub0)

## start with shrunken bounds
#diff = ub-lb
#shrink = 0.3 # percent shrink
#lb = lb + shrink*diff
#ub = ub - shrink*diff

# dimension
dim_x = len(lb)

# generate a new seed
seed = np.random.randint(int(1e6))
np.random.seed(seed)
# for data dump
outfile = f"./data/samples_{seed}.pickle"

# storage
X = np.zeros((0,dim_x)) # points
FX = np.zeros((0,dim_F)) # function values
CX = np.zeros((0,1)) # constraint values

for ii in range(max_iter):
  print("\n")
  print("iteration: ",ii)
  # sample
  Y = np.random.uniform(lb,ub,(n_points_per,dim_x))
  if ii == 0:
    # start with a feasible sample
    Y[0] = x0+1e-3*np.random.randn(dim_x) 
  FY = np.array([evaluator.eval(yy) for yy in Y])
  # compute constraint values
  CY = np.all(np.isfinite(FY),axis=1).reshape((-1,1))
  # save data
  X = np.copy(np.vstack((X,Y)))
  FX = np.copy(np.vstack((FX,FY)))
  CX = np.copy(np.vstack((CX,CY)))
  print(np.sum(CX),"Total feasible points")
  # find tightest bounds
  lb,ub = compute_bounds(X,CX)
  # ensure bounds are as large as original
  lb = np.copy(np.minimum(lb0,lb))
  ub = np.copy(np.maximum(ub0,ub))
  print('Bounds')
  print(lb,ub)
  sys.stdout.flush()
  # dump a pickle file
  outdata = {}
  outdata['X'] = X
  outdata['FX'] = FX
  outdata['CX'] = CX
  outdata['ub'] = ub
  outdata['lb'] = lb
  outdata['n_points'] = len(X)
  pickle.dump(outdata,open(outfile,"wb"))
  print('dumping',outfile)
  # enlarge
  diff = (ub-lb)/4
  ub = np.copy(ub + growth_factor*diff)
  lb = np.copy(lb - growth_factor*diff)

