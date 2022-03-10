import numpy as np
import pickle
from scipy.optimize import minimize
import sys
sys.path.append("../../utils")
sys.path.append("../../problem")
import qh_prob1
from eval_wrapper import eval_wrapper

"""
Penalty Method to solve
  min QS^2
  s.t. aspect = aspect_target
"""

# load the aspect ratio target
aspect_target = float(sys.argv[1])

# load the problem
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
prob = qh_prob1.QHProb1(vmec_input = vmec_input,aspect_target = aspect_target)
x0 = prob.x0
dim_x = prob.dim_x

# mpi rank
if prob.mpi.proc0_world:
  master = True
else:
  master = False

max_eval = 3000 # evals per iteration
gtol     = 1e-3 # stopping tolerance
max_solves = 8 # number of penalty updates
pen_param = 1.0 # penalty parameter initialization
pen_inc = 10.0 # increase parameter
ctol    = 1e-5 # target constraint tolerance
qs_gtol = 1e-3 # target gtol for qs
method  ='L-BFGS-B'
options = {'maxfun':max_eval,'gtol':gtol}

def con(xx):
  """ constraint """
  ev = prob.eval(xx)
  return ev[1]
# write the objective
def obj(xx):
  """ penalty obj """
  ev  = prob.eval(xx)
  ret = ev[0] + pen_param*ev[1]
  print(f'f(x): {ret}, qs err: {ev[0]}, con: {ev[1]}')
  return ret
def grad(xx):
  """ penalty jac """
  jac = prob.jacp(xx)
  ret = jac[0] + pen_param*jac[1]
  print('|grad|',np.linalg.norm(ret))
  return ret

# wrap the objective
func_wrap = eval_wrapper(obj,dim_x,1)

# set the seed
seed = prob.sync_seeds()

if master:
  print("Running penalty method")
  print(f"with {max_eval} evals per iteration")
  sys.stdout.flush()

for ii in range(max_solves):
  if master:
    print("\n\n\n")
    print("iteration",ii)
  res     = minimize(obj,x0,jac=grad,method=method,options=options)
  xopt = res.x
  fopt = obj(xopt)
  copt = con(xopt)
  if master:
    print("\n\n\n")
    print(res)
    print('optimal obj:',fopt)
    print('optimal con:',copt)
    sys.stdout.flush()

  # reset for next iter
  x0 = np.copy(xopt)
  if copt >ctol:
    # only increase penalty if needed
    pen_param = pen_inc*pen_param
  else:
    # check stationarity
    grad_qs = prob.jacp(xopt)[0]
    if np.linalg.norm(grad_qs) <= qs_gtol:
      break

  # get run data
  X = func_wrap.X
  FX = func_wrap.FX
  
  # dump the evals at the end
  outfilename = f"./data/data_aspect_{aspect_target}_{seed}.pickle"
  if master:
    print("\n\n\n")
    print(f"Dumping data to {outfilename}")
    outdata = {}
    outdata['xopt'] = xopt
    outdata['X'] = X
    outdata['FX'] = FX
    outdata['aspect_target'] = aspect_target
    pickle.dump(outdata,open(outfilename,"wb"))
    
