import numpy as np
import pickle
from platypus import NSGAII,Problem,Real
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
from rescale import *
import safe_eval
from eval_wrapper import eval_wrapper
from is_pareto_efficient import is_pareto_efficient


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

max_eval = int(sys.argv[1])

# set the seed
seed = np.random.randint(int(1e8))
np.random.seed(seed)


# TODO: Rescale the input variables

def objective(xx):
  """
  Optimize a non-smooth rescaled objective.
  The QS^2 upper bound is around 1e17 which would 
  introduce a huge instability into our problem if we
  rescaled by it.

  Instead we optimize the non-smooth objectives 
  [|QS|, |Aspect-A_target|]
  and rescale these instead.
  """
  return to_unit_cube(func_wrap(xx),F_lb,F_ub)

print("Running optimization")
print(f"with {max_eval} evals")
# set up NSGAII
problem = Problem(dim_x, dim_F)
for ii in range(dim_x):
  problem.types[ii] = Real(x_lb[ii],x_ub[ii])
problem.function = objective
algorithm = NSGAII(problem)
algorithm.run(max_eval)

# get run data
X = func_wrap.X
FX = func_wrap.FX

# get the non-dominated points
pareto_idx = is_pareto_efficient(FX)

# dump the evals at the end
outfilename = f"./data/NSGAII_{seed}.pickle"
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['pareto_idx'] = pareto_idx
pickle.dump(outdata,open(outfilename,"wb"))

