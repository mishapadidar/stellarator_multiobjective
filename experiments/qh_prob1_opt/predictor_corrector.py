import numpy as np
import pickle
from functools import partial
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

"""
Predictor corrector method for expanding the pareto front of
  min [Q(x), A(x)]
where Q is the mean squared quasisymmetry and A is aspect ratio.

The methods are taken from
'Gradient‑based Pareto front approximation applied
to turbomachinery shape optimization', Vasilopoulos, 2019

This script expects to be run on slurm via the batch submission script. 
If you would like to run this at the command line set `debug=True`.
"""

#####
## take inputs
#####

# choose stopping criteria
max_iter = 50 # evals per iteration
kkt_tol  = 1e-8
n_solves = 10 # number of predictor corrector solves
# target step sizes. Only one will be chosen based off of direction
aspect_step = 0.05 # positive
qs_mse_step = 5e-10 # positive

# output dir
outputdir = "../data"
if debug:
  outputdir = "./data"

# load the initial aspect ratio target
aspect_init = float(sys.argv[1])  # positive float
vmec_res = sys.argv[2] # vmec input fidelity low, mid, high
max_mode = int(sys.argv[3]) # max mode = 1,2,3,4,5...
direction = sys.argv[4] # left or right

# set direction of motion
assert direction in ['left','right']
if direction == 'left':
  qs_mse_step = 0.0
elif direction == 'right':
  aspect_step = 0.0

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
prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_init)

if debug:
  x0 = prob.x0
else:
  # find a point on the pareto front
  pareto_data = pickle.load(open(outputdir + "/pareto_optimal_points.pickle","rb"))
  idx_start = np.argmin(np.abs(pareto_data['FX'][:,0] - aspect_init))
  x0 = np.copy(pareto_data['X'][idx_start])
  del pareto_data
  # convert to higher dimension representation
  x0 = prob.increase_dimension(x0,max_mode)
  # reset just to be sure
  prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_init)
dim_x = prob.dim_x

# mpi rank
if prob.mpi.proc0_world:
  master = True
else:
  master = False

# set the seed
seed = prob.sync_seeds()

#####
## set some stuff
#####

# for reusing evals
global x_stored
global raw_stored
global jac_stored
x_stored = np.copy(x0)
raw_stored = np.copy(prob.raw(x0))
jac_stored = np.copy(prob.jacp_residuals(x0))

# initial objective values
qs_mse0 = np.mean(raw_stored[:-1]**2)
aspect0 = raw_stored[-1]

# compute initial KKT conditions
grad_qs = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ raw_stored[:-1]
grad_asp = jac_stored[-1]
lam0 = -grad_qs @ grad_asp/(grad_asp @ grad_asp)

if master:
  print('')
  print('initial aspect',aspect0)
  print('initial qs_mse',qs_mse0)
  print('initial stationary condition: ',np.linalg.norm(grad_qs + lam0*grad_asp))
  print('initial |qs grad|: ',np.linalg.norm(grad_qs))
  print('initial |aspect grad|: ',np.linalg.norm(grad_asp))

# set outfile
now     = datetime.now()
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
outfilename = outputdir + f"/data_predcorr_aspect_{aspect_init}_{barcode}.pickle"

#####
## define functions
#####

# wrap the raw objective
func_wrap = eval_wrapper(prob.raw,prob.dim_x,prob.n_qs_residuals+1)

def Objective(xx,aspect_target=aspect0,qs_mse_target=qs_mse0):
  """ Sum of squares (A-A_target)**2 + (Q-Q_target)**2 """
  # only evaluate if necessary
  if np.all(xx == x_stored):
    raw = np.copy(raw_stored)
  else:
    raw = func_wrap(xx)
  qs_mse = np.mean(raw[:-1]**2)
  aspect = raw[-1]
  ret = (aspect-aspect_target)**2 + (qs_mse - qs_mse_target)**2
  if master:
    print(f'f(x): {ret}, qs mse: {qs_mse}, aspect: {aspect}')
  return ret

def Gradient(xx,aspect_target=aspect0,qs_mse_target=qs_mse0):
  """ 
  Gradient of sum of squares function 
    G = 2*(A-A_target)*grad(A) + 2*(Q-Q_target)*grad(Q)
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  # compute the gradient
  grad_qs_mse = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ raw_stored[:-1]
  qs_mse = np.mean(raw_stored[:-1]**2)
  aspect = raw_stored[-1]
  grad_aspect = jac_stored[-1]
  grad = 2*(aspect - aspect_target)*grad_aspect + 2*(qs_mse - qs_mse_target)*grad_qs_mse

  stat_cond = np.linalg.norm(grad)
  if master:
    print('stat_cond',stat_cond)
  return np.copy(grad)

def ApproximateHessian(xx,aspect_target=aspect0,qs_mse_target=qs_mse0):
  """ 
  WARNING: this hessian is only rank-2 if aspect=aspect_target and
    qs_mse = qs_mse_target! This is a fundamental problem with the objective  
    when using < dim_x residuals. 
    The work around is to set the target values aggresively, so that
    we cannot reach them.

  Approximate Hessian of sum of squares objective
  using Gauss Newton structure of the objective and Q.
  The exact hessian is 
    H = 2*outer(grad(A),grad(A)) + 2*(A-A_target)*Hess(A)
      + 2*outer(grad(Q),grad(Q)) + 2*(Q - Q_target)*Hess(Q)
  In a Gauss-Newton fashion we assume A is linear, i.e.
    Hess(A) = 0
  Q is a sum of squares, 
    Q = (1/n_qs_residuals)sum q_i^2 
  and so Q itself is ammenable Gauss Newton approximation:
    Hess(Q) = (2/n_qs_residuals)(J_q)^T(J_q)
  where J_q is the jacobian of [q_1,...].
  We return an approximate hessian of the form
    H = 2*outer(grad(A),grad(A)) + 2*outer(grad(Q),grad(Q)) 
        + (4/n_qs_residuals)*(Q - Q_target)*(J_q)^T(J_q)
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  # compute grad(Q)
  grad_qs_mse = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ raw_stored[:-1]
  # compute Q
  qs_mse = np.mean(raw_stored[:-1]**2)
  # compute grad(A)
  grad_aspect = jac_stored[-1]
  # compute A
  aspect = raw_stored[-1]

  # compute a diagonal hessian for aspect
  hess_aspect = np.zeros(prob.dim_x)
  h_fd = 1e-6
  E = np.eye(prob.dim_x)
  for jj in range(prob.dim_x):
    # central difference
    cd = prob.aspect(xx + h_fd*E[jj]) - 2*aspect + prob.aspect(xx - h_fd*E[jj])
    cd = cd/h_fd/h_fd
    hess_aspect[jj] = cd
  hess_aspect = (aspect-aspect_target)*np.diag(hess_aspect)

  # qs_mse hessian
  hess_qs_mse = (2/prob.n_qs_residuals)*(qs_mse - qs_mse_target)*jac_stored[:-1].T @ jac_stored[:-1]

  H = 2*np.outer(grad_aspect,grad_aspect) + 2*np.outer(grad_qs_mse,grad_qs_mse)\
      + 2*hess_qs_mse + 2*hess_aspect
  return np.copy(H)

def PredictorStep(xx,target_xx,target_new):
  """ 
  Compute the predictor step.

  Suppose that x0 solves 
    (star) min_x (A - A_target)**2 + (Q - Q_target)**2 
  then 
    0 = (A - A_target)*grad(A) + (Q - Q_target)*grad(Q)
  By defining Target0 = [A_target,Q_target], we can implicitly differentiate
  the optimality condition with respect to Target. The result is the system
    J.T = B @ D_T(x)
  where J is the (2,dim_x) jacobian J = [[grad(A)],[grad Q]], D_T(x) is the derivative
  of the solution x with respect to Target, and B is the (dim_x,dim_x) hessian matrix.
    B = outer(grad(A),grad(A)) + (A-A_target)*Hess(A)
      + outer(grad(Q),grad(Q)) + (Q - Q_target)*Hess(Q)
  The do the predictor step by using the taylor expansion
    x = x0 + B^{-1} @ J.T @ (Target - Target0)
  where B^{-1} and J are evaluated at (x0,Target0).

  We use the Gauss-Newton approximation of B, exactly as shown in the ApproximateHessian() 
  method.

  input:
  xx: pareto optimal point, or point satisfying (star) with 
     respect to a point which dominates the pareto front.
  target_xx: the target with respect to xx which satifies (star)
  target_new: a new target, which is a small deviation from target_xx

  return:
  yy: (dim_x,) array, an estimate of the solution of (star) with respect to target
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  # compute B
  B = ApproximateHessian(xx,*target_xx)
  Q,R = np.linalg.qr(B)
  # compute the jacobian
  grad_qs_mse = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ raw_stored[:-1]
  grad_aspect = jac_stored[-1]
  J = np.vstack((grad_aspect,grad_qs_mse)) # rows are gradients (2,dim_x)

  # check the conditioning
  hess_cond = np.linalg.cond(B)
  if master:
    print('Hessian condition number: ',hess_cond)
  
  # compute prediction
  if hess_cond < 1e14:
    yy = xx + np.linalg.solve(R, Q.T @ J.T @ (target_new-target_xx))
  else:
    # if poorly conditioned just return xx 
    yy = xx
  return np.copy(yy)

#####
# run predictor/corrector iteration
#####

# set initial target
target_k = np.array([aspect0,qs_mse0]) - np.array([aspect_step,qs_mse_step])
aspect_target = target_k[0]
qs_mse_target = target_k[1]

x_k = np.copy(x0)
for ii in range(n_solves):
  if master:
    print("")
    print("="*80)
    print('iteration',ii)
    print('aspect_target',aspect_target)
    print('qs_mse_target',qs_mse_target)
    print("")
    print("running newton method")
    print(f"for {max_iter} steps or stationary target {kkt_tol}")
    sys.stdout.flush()
  
  # wrap the functions
  OptObjective = partial(Objective,**{'aspect_target':aspect_target,'qs_mse_target':qs_mse_target})
  OptGradient = partial(Gradient,**{'aspect_target':aspect_target,'qs_mse_target':qs_mse_target})
  OptHessian = partial(ApproximateHessian,**{'aspect_target':aspect_target,'qs_mse_target':qs_mse_target})
  
  # run the corrector
  xopt = NewtonLinesearch(OptObjective,OptGradient,OptHessian,x_k,max_iter=max_iter,gtol=kkt_tol)

  # compute diagnostics
  rawopt = prob.raw(xopt)
  jacopt = prob.jacp_residuals(xopt)
  grad_qs = (2/prob.n_qs_residuals)*jacopt[:-1].T @ rawopt[:-1]
  grad_asp = jacopt[-1]
  qs_mse_opt = np.mean(rawopt[:-1]**2)
  aspect_opt = rawopt[-1]
  copt = aspect_opt - aspect_target
  # check pareto optimality conditions
  lam = -(grad_qs @ grad_asp)/(grad_asp @ grad_asp)
  stat_cond = np.linalg.norm(grad_qs + lam*grad_asp)

  if master:
    print("")
    print('optimal qs mse:',qs_mse_opt)
    print('aspect',aspect_opt)
    print('optimal con:',copt)
    print('lagrange multiplier: ',lam)
    print('norm stationary cond: ',stat_cond)

  # set a new target
  target_kp1 = np.copy(target_k - np.array([aspect_step,qs_mse_step]))

  # compute predictor step
  x_kp1 = PredictorStep(xopt,target_k,target_kp1)
  
  # check objective and gradient
  if master:
    print("")
    print("Predictor step")
    print("New Target",target_kp1)
    sys.stdout.flush()

  # compute to print performance and save eval
  Objective(x_kp1,*target_kp1)
  Gradient(x_kp1,*target_kp1)
  
  if master:
    # dump the evals at the end
    print("")
    print(f"Dumping data to {outfilename}")
    outdata = {}
    outdata['dim_x'] = dim_x
    outdata['max_mode'] = max_mode
    outdata['xopt'] = xopt
    outdata['lam'] = lam # lagrange multiplier
    outdata['rawopt'] = rawopt
    outdata['qs_mse_opt'] = qs_mse_opt
    outdata['aspect_opt'] = aspect_opt
    outdata['copt'] = copt
    outdata['aspect_target'] = aspect_target
    outdata['qs_mse_target'] = aspect_target
    outdata['jacopt'] = jacopt 
    outdata['X'] = func_wrap.X
    outdata['RawX'] = func_wrap.FX
    outdata['kkt_tol'] = kkt_tol
    outdata['stationary_condition'] = stat_cond
    outdata['grad_aspect'] = grad_asp
    if not debug:
      pickle.dump(outdata,open(outfilename,"wb"))

  # reset for next iteration
  x_k = np.copy(x_kp1)
  target_k = np.copy(target_kp1)
  aspect_target = target_k[0]
  qs_mse_target = target_k[1]
  

