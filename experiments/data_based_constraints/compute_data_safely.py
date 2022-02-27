import numpy as np
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
import safe_eval
import pickle
import time

"""
Compute the data based bound constraints with safe evaluation.
This script can only be run serially, i.e. with 
`python3 compute_data_safely.py`

This script must be warm started via a pickle file, such as the 
output of running `compute_data.py`
"""

# start up a safe evaluator
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
eval_script = "./qh_prob1_safe_eval.py"
dim_F = 2
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script)

# warm start with a pickle file
load_file = "samples_374972.pickle"
# parameters
max_iter = 20
# number of points per iteration
n_points_per = 100 # need more than 1
# growth factor ( greater than 0)
growth_factor = 1.5

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

# warm start the bounds from a pickle file
indata = pickle.load(open(load_file,"rb"))
lb,ub = compute_bounds(indata['X'],indata['CX'])
dim_x = len(lb)

# generate a new seed
seed = np.random.randint(int(1e6))
np.random.seed(seed)
# for data dump
outfile = f"samples_{seed}.pickle"

# storage
X = np.zeros((0,dim_x)) # points
FX = np.zeros((0,dim_F)) # function values
CX = np.zeros((0,1)) # constraint values

for ii in range(max_iter):
  print("\n")
  print("iteration: ",ii)
  # sample
  Y = np.random.uniform(lb,ub,(n_points_per,dim_x))
  FY = np.array([evaluator.eval(yy) for yy in Y])
  # compute constraint values
  CY = (FY[:,0] != np.inf).reshape((-1,1))
  # save data
  X = np.copy(np.vstack((X,Y)))
  FX = np.copy(np.vstack((FX,FY)))
  CX = np.copy(np.vstack((CX,CY)))
  # find tightest bounds
  lb_kp1,ub_kp1 = compute_bounds(X,CX)
  # ensure bound growth
  lb = np.copy(np.minimum(lb,lb_kp1))
  ub = np.copy(np.maximum(ub,ub_kp1))
  # dump a pickle file
  outdata = {}
  outdata['X'] = X
  outdata['FX'] = FX
  outdata['CX'] = CX
  outdata['ub'] = ub
  outdata['lb'] = lb
  outdata['n_points'] = len(X)
  pickle.dump(outdata,open(outfile,"wb"))
  # enlarge
  diff = (ub-lb)/4
  ub = np.copy(ub + growth_factor*diff)
  lb = np.copy(lb - growth_factor*diff)

