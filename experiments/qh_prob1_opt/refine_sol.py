import numpy as np
import pickle
from datetime import datetime
import sys
sys.path.append("../../../utils")
sys.path.append("../../../optim")
sys.path.append("../../../problem")

debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../optim")
  sys.path.append("../../problem")
import qh_prob1
from finite_difference import finite_difference_hessian
from eval_wrapper import eval_wrapper
from newton_linesearch import NewtonLinesearch
from find_warm_start import find_warm_start

"""
Refine a solution to high fidelity by solving the KKT
system for an equality constrained program.

We solve
     min Q(x) := 1/N sum_i q_i^2(x)
s.t. A(x) = A*

with a Newton method. Q is the mean squared quasisymmetry error
and A is the aspect ratio.
"""

#####
## take inputs
#####

max_iter = 100 # evals per iteration
kkt_tol  = 1e-8

# load the aspect ratio target
aspect_target = float(sys.argv[1])  # positive float
outputdir = sys.argv[2] # should be formatted as i.e. "../data"
vmec_res = sys.argv[3] # vmec input fidelity low, mid, high
max_mode = int(sys.argv[4]) # max mode = 1,2,3,4,5...

assert max_mode <=5, "max mode out of range"
assert vmec_res in ["low","mid","high"]
if vmec_res == "low":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start"
  if debug:
    vmec_input = "../../problem/input.nfp4_QH_warm_start"
elif vmec_res == "mid":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start_mid_res"
elif vmec_res == "high":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start_high_res"
# load the problem
prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)

if debug:
  x0 = prob.x0
else:
  # warm start
  dir_list = ["../data"]
  if debug:
    dir_list = ["./data"]
  # find a good starting point
  x0 = find_warm_start(aspect_target,dir_list,thresh=5e-4)
  # convert to higher dimension representation
  x0 = prob.increase_dimension(x0,max_mode)
  # reset just to be sure
  prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)
dim_x = prob.dim_x

#####
## set some stuff
#####

# mpi rank
if prob.mpi.proc0_world:
  master = True
else:
  master = False

# set outfile
now     = datetime.now()
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
outfilename = outputdir + f"/data_aspect_{aspect_target}_{barcode}.pickle"

# set the seed
seed = prob.sync_seeds()

# lagrange multiplier initialization
raw0 = prob.raw(x0)
qs_mse0 = np.mean(raw0[:-1]**2)
aspect0 = raw0[-1]
global jac_reuse
jac_reuse = prob.jacp_residuals(x0)
grad_qs = (2/prob.n_qs_residuals)*jac_reuse[:-1].T @ raw0[:-1]
grad_asp = jac_reuse[-1]
lam0 = -grad_qs @ grad_asp/(grad_asp @ grad_asp)

if master:
  print('')
  print('initial qs_mse',qs_mse0)
  print('initial aspect',aspect0)
  print('aspect target: ', aspect_target)
  print('initial stationary condition: ',np.linalg.norm(grad_qs + lam0*grad_asp))
  print('initial |qs grad|: ',np.linalg.norm(grad_qs))
  print('initial |aspect grad|: ',np.linalg.norm(grad_asp))
  print('kkt tol:',kkt_tol)


#####
## define functions
#####

# wrap the raw objective
func_wrap = eval_wrapper(prob.raw,prob.dim_x,prob.n_qs_residuals+1)

# write the objective
def LagrangeGradient(yy,h=1e-7):
  """ 
  Gradient of the Lagrangian 
  G = [grad(qs_mse) + lam*grad(aspect),aspect-aspect_target]
  """
  # split
  xx = np.copy(yy[:-1])
  lam = yy[-1]
  # gradients
  raw = prob.raw(xx)
  global jac_reuse
  jac_reuse = prob.jacp_residuals(xx,h=h)
  grad_qs = (2/prob.n_qs_residuals)*jac_reuse[:-1].T @ raw[:-1]
  grad_asp = jac_reuse[-1]
  # grad_x L
  grad_x = grad_qs + lam*grad_asp
  # grad_l L
  grad_l = prob.aspect(xx) - aspect_target
  grad = np.append(grad_x,grad_l)
  # measure stationarity
  stat_cond = np.linalg.norm(grad_x)
  if master:
    print(f'stationary cond: {stat_cond}, qs mse: {np.mean(raw[:-1]**2)}, aspect: {raw[-1]}')
  return grad
def LagrangeHessian(yy,h=1e-5):
  """ Approximation Hessian of the Lagrangian 
  True hessian is
  H = [[Hess(qs_mse) + lam*Hess(aspect), grad(aspect)],
       [grad(aspect).T,                  0           ]]
  We aproximate Hess(qs_mse) with a Gauss Newton approximation
  """
  # split
  dim_y = len(yy)
  xx = np.copy(yy[:-1])
  lam = yy[-1]

  # reuse the jacobian
  grad_asp = jac_reuse[-1]
  # gauss newton hessian of qs_mse
  hess_qs_mse = jac_reuse[:-1].T @ jac_reuse[:-1]/prob.n_qs_residuals
  # direct hessian of aspect
  hess_asp = finite_difference_hessian(prob.aspect,xx,h=h)

  # stack it
  hess = np.zeros((dim_y,dim_y))
  hess[:dim_x,:dim_x] = np.copy(hess_qs_mse + lam*hess_asp)
  hess[:dim_x,-1] = grad_asp
  hess[-1,:dim_x] = grad_asp

  return hess

#####
## run newton method
#####

if master:
  print("")
  print("Running newton method")
  print(f"for {max_iter} steps or stationary target {kkt_tol}")
  sys.stdout.flush()

# run it
y0 = np.append(x0,lam0)
yopt = NewtonLinesearch(LagrangeGradient,LagrangeHessian,y0,max_iter=max_iter,ftarget=kkt_tol)
xopt = yopt[:-1]
lamopt = yopt[-1]

# compute diagnostics
rawopt = prob.raw(xopt)
jacopt = prob.jacp_residuals(xopt)
grad_qs = (2/prob.n_qs_residuals)*jacopt[:-1].T @ rawopt[:-1]
grad_asp = jacopt[-1]
qs_mse_opt = np.mean(rawopt[:-1]**2)
aspect_opt = rawopt[-1]
copt = aspect_opt - aspect_target
stat_cond = np.linalg.norm(grad_qs + lamopt * grad_asp)

if master:
  print("")
  print('optimal qs mse:',qs_mse_opt)
  print('aspect',aspect_opt)
  print('optimal con:',copt)
  print('lagrange multiplier: ',lamopt)
  print('norm stationary cond: ',stat_cond)
  sys.stdout.flush()

  # dump the evals at the end
  print("")
  print(f"Dumping data to {outfilename}")
  outdata = {}
  outdata['dim_x'] = dim_x
  outdata['max_mode'] = max_mode
  outdata['xopt'] = xopt
  outdata['lamopt'] = lamopt # lagrange multiplier
  outdata['rawopt'] = rawopt
  outdata['qs_mse_opt'] = qs_mse_opt
  outdata['aspect_opt'] = aspect_opt
  outdata['copt'] = copt
  outdata['aspect_target'] = aspect_target
  outdata['jacopt'] = jacopt 
  outdata['X'] = func_wrap.X
  outdata['RawX'] = func_wrap.FX
  outdata['kkt_tol'] = kkt_tol
  outdata['stationary_condition'] = stat_cond
  outdata['grad_aspect'] = grad_asp
  if not debug:
    pickle.dump(outdata,open(outfilename,"wb"))
  

