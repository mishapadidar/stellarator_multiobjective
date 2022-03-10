import numpy as np
import pickle
from scipy.optimize import minimize
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
import safe_eval
from rescale import *
from eval_wrapper import eval_wrapper
from is_pareto_efficient import is_pareto_efficient

"""
MO optimization via Tchebycheff scalarization
"""

# load the bound constraints
bounds = pickle.load(open("../../problem/bounds.nfp4_QH_warm_start_high_res.pickle","rb"))
x_lb,x_ub,F_lb,F_ub = bounds['lb'],bounds['ub'],bounds['F_lb'],bounds['F_ub']
# set F_lb to the analytic lower bound
F_lb = 0.0*F_lb

# sizes
dim_x = len(x_lb)
dim_F = len(F_lb)

# start up a safe evaluator
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
eval_script = "./qh_prob1_safe_eval.py"
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script)

# wrap the evaluator
func_wrap = eval_wrapper(evaluator.eval,dim_x,dim_F)

# maximum evaluations
max_eval = 3000

# load the scalarization weights
weight1 = float(sys.argv[1])
assert (0.0 <= weight1 and weight1 <= 1.0), "Weight must be between 0 and 1"
weights = np.array([weight1,1.0-weight1])

# set the seed
seed = np.random.randint(int(1e8))
np.random.seed(seed)

# rescale the objectives
def objective(xx):
  """ Tchebycheff objective. 
      F_lb = 0 is the ideal point. 
      inputs and objectives are scaled to unit cube.
  """
  # map back from unit cube
  xx = np.copy(from_unit_cube(xx,x_lb,x_ub))
  # map objectives to unit cube
  ev = to_unit_cube(func_wrap(xx),F_lb,F_ub)
  print(ev)
  sys.stdout.flush()
  # compute tchebycheff
  ret = np.max(weights*ev)
  return ret

# TODO: load x0 through the cmd line
# Generate a feasible x0
print("Generating x0")
valid = False
n_attempts = 0
while not valid:
  x0 = np.random.uniform(x_lb,x_ub)
  n_attempts += 1
  print("attempt: ",n_attempts)
  sys.stdout.flush()
  if np.all(np.isfinite(evaluator.eval(x0))):
    valid = True
  elif n_attempts == 100:
    print("\n\n")
    print("Cant find a feasible starting point")
    sys.stdout.flush()
    quit()
# map to cube
x0 = to_unit_cube(x0,x_lb,x_ub)

print("Running optimization")
print(f"with {max_eval} evals")
print(f"and weights {weights}")
sys.stdout.flush()
method='Nelder-Mead'
options = {'maxfev':max_eval,'xatol':1e-3}
res = minimize(objective,x0,method=method,options=options)
print(res)
sys.stdout.flush()

# get run data
X = func_wrap.X
FX = func_wrap.FX

# get the non-dominated points
pareto_idx = is_pareto_efficient(FX)

# dump the evals at the end
outfilename = f"./data/data_tcheby_{seed}.pickle"
print(f"Dumping data to {outfilename}")
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['pareto_idx'] = pareto_idx
outdata['weights'] = weights
outdata['method'] = method
pickle.dump(outdata,open(outfilename,"wb"))

