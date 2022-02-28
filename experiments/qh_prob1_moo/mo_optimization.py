import numpy as np
import pickle
from platypus import NSGAII,Problem,Real
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
import safe_eval
from eval_wrapper import eval_wrapper
from is_pareto_efficient import is_pareto_efficient

# starting point
x0_input = "../../problem/x0.nfp4_QH_warm_start_high_res.pickle"
x0 = pickle.load(open(x0_input,"rb"))
dim_x = len(x0)

# start up a safe evaluator
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
eval_script = "./qh_prob1_safe_eval.py"
dim_F = 2
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script)
# wrap the evaluator
func_wrap = eval_wrapper(evaluator.eval,dim_x,dim_F)

# set the seed
seed = np.random.randint(int(1e8))
np.random.seed(seed)

# TODO: run without bounds... they may be over constraining
# load the bound constraints
bounds = pickle.load(open("../../problem/bounds.nfp4_QH_warm_start_high_res.pickle","rb"))
lb,ub,F_lb,F_ub = bounds['lb'],bounds['ub'],bounds['F_lb'],bounds['F_ub']

# rescale the objectives
def objective(xx):
  return (func_wrap(xx) - F_lb)/(F_ub - F_lb)

# set up NSGAII
max_eval = 10000
problem = Problem(dim_x, dim_F)
for ii in range(dim_x):
  problem.types[ii] = Real(lb[ii],ub[ii])
problem.function = objective
algorithm = NSGAII(problem)
algorithm.run(max_eval)

# get run data
X = func.X
FX = func.FX

# get the non-dominated points
pareto_idx = is_pareto_efficient(FX)

# dump the evals at the end
outfilename = f"./data/NSGAII_{seed}.pickle"
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['pareto_idx'] = pareto_idx
pickle.dump(outdata,open(outfilename,"wb"))

