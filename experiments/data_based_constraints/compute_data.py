import numpy as np
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
import safe_eval
import pickle
import time

# load the problem and starting point
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
x0_input = "../../problem/x0.nfp4_QH_warm_start_high_res.pickle"
x0 = pickle.load(open(x0_input,"rb"))
dim_x = len(x0)

# start up a safe evaluator
dim_F = 2
n_partitions = 2 # one less than the number in the slurm submit file
eval_script = "./qh_prob1_safe_eval.py"
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script,n_partitions)

# warm start with a pickle file (o/w set to None)
load_file = "samples_610834.pickle"
#load_file = None
# parameters
max_iter = 2
# number of points per iteration
n_points_per = 6 # need more than 1
n_points = max_iter*n_points_per
# growth factor ( greater than 0)
growth_factor = 1.5
# initial box size
max_pert = 0.001 
ub = x0 + max_pert
lb = x0 - max_pert

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

# load a pickle file
if load_file is not None:
  indata = pickle.load(open(load_file,"rb"))
  X = indata['X']
  FX = indata['FX']
  CX = indata['CX']
  # compute bounds
  lb,ub = compute_bounds(X,CX)
  # keep same output data file name
  outfile = load_file
else:
  seed = np.random.randint(int(1e6))
  np.random.seed(seed)
  # storage
  X = np.zeros((0,dim_x)) # points
  FX = np.zeros((0,dim_F)) # function values
  CX = np.zeros((0,1)) # constraint values
  # for data dump
  outfile = f"samples_{seed}.pickle"


for ii in range(max_iter):
  print("\n\n\n")
  print("iteration: ",ii)
  # sample
  Y = np.random.uniform(lb,ub,(n_points_per,dim_x))
  t0 = time.time()
  FY = evaluator.evalp(Y)
  print(f"{time.time() - t0} seconds for evaluations")
  # compute constraint values
  CY = (FY[:,0] != np.inf).reshape((-1,1))
  # save data
  X = np.copy(np.vstack((X,Y)))
  FX = np.copy(np.vstack((FX,FY)))
  CX = np.copy(np.vstack((CX,CY)))
  # find tightest bounds
  lb,ub = compute_bounds(X,CX)
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

