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
from eval_wrapper import eval_wrapper
from gauss_newton import GaussNewton
from block_coordinate_gauss_newton import BlockCoordinateGaussNewton

"""
Predictor-Corrector method for local exploration of the Pareto
front. 

The predictor expansion uses the KKT conditions for the
epsilon-constraint method.
The corrector step solves the epsilon-constraint problem
using a penalty approach.
"""

#####
## take inputs
#####

# predictor corrector steps
max_pc_steps = 10
# penalty method parameters
max_solves = 2 # number of penalty updates
pen0    = 1e4 # initial penalty param.
pen_inc = 10.0 # increase parameter
ctol    = 1e-6 # target constraint tolerance
# Gauss-Newton parameters
max_iter = 20 # evals per iteration
ftarget  = 1e-11
ftol_abs = ftarget*1e-5
kkt_tol  = 1e-7 

# load the initial aspect ratio target
aspect_target = float(sys.argv[1])  # positive float
vmec_res = sys.argv[2] # vmec input fidelity low, mid, high
max_mode = int(sys.argv[3]) # max mode = 1,2,3,4,5...
aspect_step = float(sys.argv[4]) # can be negative for left expansion

assert max_mode <=5, "max mode out of range"
assert vmec_res in ["low","mid","high","super"]
if vmec_res == "low":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start"
  if debug:
    vmec_input = "../../problem/input.nfp4_QH_warm_start"
elif vmec_res == "mid":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start_mid_res"
elif vmec_res == "high":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start_high_res"
elif vmec_res == "super":
  vmec_input = "../../../problem/input.nfp4_QH_warm_start_super_high_res"
# load the problem
prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)

# output directory
outputdir = "../data"
if debug:
  outputdir = "./data"

# choose starting point
if debug:
  x0 = prob.x0
  aspect_target = prob.aspect(x0)
else:
  # find a point on the pareto front
  pareto_data = pickle.load(open(outputdir + "/pareto_optimal_points.pickle","rb"))
  idx_start = np.argmin(np.abs(pareto_data['FX'][:,0] - aspect_target))

  # choose the point 
  x0 = np.copy(pareto_data['X'][idx_start])
  # reset the aspect target to match current point
  aspect_target = pareto_data['FX'][:,0][idx_start]

  del pareto_data
  # convert to higher dimension representation
  x0 = prob.increase_dimension(x0,max_mode)

  # reset with new aspect target
  prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)
dim_x = prob.dim_x

# mpi rank
if prob.mpi.proc0_world:
  master = True
else:
  master = False

if master:
  print("Running Predictor-Corrector Penalty method")
  print(f"with {aspect_step} aspect_step")
  print(f"with {max_solves} penalty solves")
  print(f"and {max_iter} Gauss-Newton steps per iteration")
  print('ftarget:',ftarget)
  print('ftol_abs:',ftol_abs)
  print('kkt tol:',kkt_tol)
  sys.stdout.flush()

#####
## set some stuff
#####

# set outfile
now     = datetime.now()
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
outfilename = outputdir + f"/data_aspect_{aspect_target}_{barcode}.pickle"

# set the seed
seed = prob.sync_seeds()


#####
## initialize penalty parameter
#####

# for reusing evals
global x_stored
global raw_stored
global jac_stored
x_stored = np.copy(x0)
raw_stored = np.copy(prob.raw(x0))
jac_stored = np.copy(prob.jacp_residuals(x0))

def computeObjectives(xx):
  """
  compute aspect and qs_mse
  """
  global x_stored
  global raw_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    raw = np.copy(prob.raw(xx))
  else:
    raw = np.copy(raw_stored)
  
  qs_mse = np.mean(raw[:-1]**2)
  aspect = raw[-1]
  return aspect,qs_mse

def computeGradients(xx):
  """
  Compute the gradients at xx
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  grad_qs = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ raw_stored[:-1]
  grad_asp = jac_stored[-1]
  return np.copy(grad_asp),np.copy(grad_qs)

def computeLagrange(xx):
  """
  Compute the Lagrange multiplier for the epsilon-constraint
  problem.
  
  lam satisfies
    grad(Q) + lam*grad(A) = 0
    lam >= 0
    lam*(A-A_target) = 0
    A <= A_target

  Complementary slackness maintains that A must exactly equal A_target
  in order for lam to be non-zero. So in practice, we perturb A_target to
  equal A(xx), and solve for the lagrange multiplier of the slightly perturbed
  problem.

  Moreover, we set lam by solving
    min |grad(Q) + lam*grad(A)| subject to lam >= 0
  which has solution
    lam = max(-grad(Q) @ grad(A)/|grad(A)|^2,0)
  
  input: xx, point, 1d array
  """
  grad_asp,grad_qs = computeGradients(xx)
  # compute kkt conditions
  lam = max(-(grad_qs @ grad_asp)/(grad_asp @ grad_asp),0.0)
  return lam
def StationaryCondition(xx):
  """
  Compute the violation of the stationary condition in 2-norm
  """
  grad_asp,grad_qs = computeGradients(xx)
  lam = computeLagrange(xx)
  # compute stationary condition
  stat_cond = np.linalg.norm(grad_qs + lam*grad_asp)
  return stat_cond
def setPenParam(xx,pen0):
  """
  Set the penalty parameter.

  xx: point, 1d-array
  pen0: initial penalty parameter, as a ratio |grad(Q)|/|grad(A)|
       i.e. if pen0 = 1 then |grad(Q)|, |grad(A)| will have the same
       weight in the penalty gradient
  """
  grad_asp,grad_qs = computeGradients(xx)
  # pen parameter initialization
  gc_ratio = np.linalg.norm(grad_qs)/np.linalg.norm(grad_asp)
  # increase the penalty param
  pen_param = pen0*gc_ratio
  return pen_param

# initial objectives and gradients
aspect0,qs_mse0 = computeObjectives(x0)
grad_asp,grad_qs = computeGradients(x0)
lam = computeLagrange(x0)
pen_param = setPenParam(x0,pen0)
stat_cond = StationaryCondition(x0)

if master:
  print('')
  print("initial diagonstics")
  print('initial aspect',aspect0)
  print('initial qs_mse',qs_mse0)
  print('initial penalty parameter: ',pen_param)
  print('initial |qs grad|: ',np.linalg.norm(grad_qs))
  print('initial |aspect grad|: ',np.linalg.norm(grad_asp))
  print('initial stat_cond:',stat_cond)
  print('initial lagrange multiplier:',lam)

#####
## define functions
#####

# wrap the raw objective
func_wrap = eval_wrapper(prob.raw,prob.dim_x,prob.n_qs_residuals+1)

# TODO:explicitly include aspect_target in args
def Constraint(xx):
  """ constraint c(x) <= 0"""
  ev = prob.aspect(xx) - aspect_target
  return ev
# TODO:explicitly include aspect_target, pen_param in args
def PenaltyResiduals(xx):
  """ weighted penalty residuals 
    [(1/sqrt(m))q1(x),...,(1/sqrt(m))qm(x),sqrt(pen)*max(A(x) - A*,0)]
  """
  # compute raw values
  #resid = prob.raw(xx)
  resid = func_wrap(xx)
  # for print
  qs_mse = np.mean(resid[:-1]**2)
  asp = resid[-1] 
  # compute residual
  resid[-1] = max(resid[-1] - aspect_target,0)
  # weight the residuals
  resid[:-1] *= np.sqrt(1.0/prob.n_qs_residuals)
  resid[-1]  *= np.sqrt(pen_param)
  ff = np.sum(resid**2)
  if master:
    print(f'f(x): {ff}, qs mse: {qs_mse}, aspect: {asp}')
  return resid
# TODO:explicitly include aspect_target, pen_param in args
def PenaltyObjective(xx):
  """ penalty obj 
  p(x) = Q(x) + pen*max(0,(A(x) - A*))**2   
  """
  ret = np.sum(PenaltyResiduals(xx)**2)
  return ret
# TODO:explicitly include aspect_target, pen_param in args
def JacPenaltyResiduals(xx,h=1e-7):
  """Weighted penalty residuals jacobian 
     Method is comptabile with 
     BlockCoordinateGaussNewton

     Jacobian of 
     [(1/sqrt(m))q1(x),...,(1/sqrt(m))qm(x),sqrt(pen)*max(A(x) - A*,0)]
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  jac = np.copy(jac_stored)
  # make sure to take gradient of max
  asp = raw_stored[-1]
  if asp -aspect_target <= 0.0:
    jac[-1] = 0.0
  # weight the residuals
  jac[:-1] *= np.sqrt(1.0/prob.n_qs_residuals)
  jac[-1] *= np.sqrt(pen_param)
  if master:
    print('computing jac')
  return jac

def AspectHessian(xx,h=1e-5):
  """
  Compute a diagonal hessian for aspect using central differences.
  xx: point, 1d array.
  """
  hess_aspect = np.zeros(prob.dim_x)
  E = np.eye(prob.dim_x)
  # compute A
  aspect = prob.aspect(xx)
  for jj in range(prob.dim_x):
    # central difference
    cd = prob.aspect(xx + h*E[jj]) - 2*aspect + prob.aspect(xx - h*E[jj])
    hess_aspect[jj] = cd/h/h
  hess_aspect = np.diag(hess_aspect)
  return np.copy(hess_aspect)

def PredictorStep(xx,aspect_target,aspect_step):
  """
  Compute the Predictor step.

  Let x be a solution to the epsilon-constraint problem. Assume x satisfies
  A(x) = A_target exactly. If not, then perturb A_target slightly so that it
  does. Then we can expand the solution of the epsilon-constraint problem
  around x and the lagrange multiplier lam. The derivatives with respect
  to A_target, [Dx, dlam], satisfy
    [[hess(Q) + lam*hess(A), grad(A)],[grad(A).T, 0]] @ [Dx, dlam] = [0, 1]
  Then the expansion is 
    x(A_target + A_step) = x(A_target) + Dx(A_target) * A_step
  """
  global x_stored
  global raw_stored
  global jac_stored

  # only evaluate if necessary
  if np.any(xx != x_stored):
    x_stored = np.copy(xx)
    raw_stored = np.copy(prob.raw(xx))
    jac_stored = np.copy(prob.jacp_residuals(xx))

  # lagrange multiplier
  lam = computeLagrange(xx)
  # aspect gradient 
  grad_aspect = jac_stored[-1]
  # aspect hessian
  hess_aspect = AspectHessian(xx)
  # qs_mse hessian (Gauss-Newton approximation)
  hess_qs_mse = (2/prob.n_qs_residuals)*jac_stored[:-1].T @ jac_stored[:-1]
  # construct B
  grad_aspect_pad = np.append(grad_aspect,0.0)
  B = np.hstack((hess_qs_mse + lam*hess_aspect,np.reshape(grad_aspect,(-1,1))))
  B = np.vstack((B,grad_aspect_pad))
  # solve for Dx
  rhs = np.zeros(prob.dim_x+1)
  rhs[-1] = 1.0
  Dx = np.linalg.solve(B,rhs)[:-1]

  # expand 
  x_new = xx + aspect_step*Dx
  # take a step from the true current aspect
  aspect_new = prob.aspect(xx) + aspect_step
  return np.copy(x_new),aspect_new

#####
## run penalty method
#####


for n_pc_step in range(max_pc_steps):
  if master:
    print("\n")
    print("="*80)
    print("Predictor-Corrector iteration",n_pc_step)

  # set the new x0 and aspect_target
  x0,aspect_target = PredictorStep(x0,aspect_target,aspect_step)
  aspect0,qs_mse0 = computeObjectives(x0)
  if master:
    print(f"aspect target {aspect_target}")
    print(f"predicted point aspect {aspect0}, qs_mse {qs_mse0}")
  
  # reset the penalty parameter
  pen_param = setPenParam(x0,pen0)

  for n_pen_step in range(max_solves):
    if master:
      print("")
      print("penalty iteration",n_pen_step)
    xopt = GaussNewton(PenaltyResiduals,JacPenaltyResiduals,x0,max_iter=max_iter,ftarget=ftarget,ftol_abs=ftol_abs,gtol=kkt_tol)
    fopt = PenaltyObjective(xopt)
    copt = Constraint(xopt)
    if master:
      print("")
      print('optimal obj:',fopt)
      print('optimal con:',copt)
      print('pen param:',pen_param)
      sys.stdout.flush()
   
    # compute diagnostics
    lam = computeLagrange(xopt)
    stat_cond = StationaryCondition(xopt)
    grad_asp,grad_qs = computeGradients(xopt)
    aspect_opt,qs_mse_opt = computeObjectives(xopt)
    # values for saving
    rawopt = prob.raw(xopt)
    jacopt = np.copy(jac_stored)

    # check KKT conditions
    if copt <= ctol and stat_cond <=kkt_tol:
      KKT = True
    else:
      KKT = False

    if master:
      print('qs mse',qs_mse_opt)
      print('aspect',aspect_opt)
      print('stationary cond: ',grad_qs + lam*grad_asp)
      print('lagrange multiplier: ',lam)
      print('norm stationary cond: ',stat_cond)
      sys.stdout.flush()
  
  
    # dump the evals at the end
    if master:
      print("")
      print(f"Dumping data to {outfilename}")
      outdata = {}
      outdata['dim_x'] = dim_x
      outdata['max_mode'] = max_mode
      outdata['xopt'] = xopt
      outdata['rawopt'] = rawopt
      outdata['qs_mse_opt'] = qs_mse_opt
      outdata['aspect_opt'] = aspect_opt
      outdata['copt'] = copt
      outdata['aspect_target'] = aspect_target
      outdata['ctol'] = ctol
      outdata['jacopt'] = jacopt 
      outdata['X'] = func_wrap.X
      outdata['RawX'] = func_wrap.FX
      outdata['pen_param'] = pen_param
      outdata['KKT'] = KKT
      outdata['kkt_tol'] = kkt_tol
      outdata['stationary_condition'] = stat_cond
      outdata['lam'] = lam # lagrange multiplier
      outdata['grad_qs'] = grad_qs
      outdata['grad_aspect'] = grad_asp
      pickle.dump(outdata,open(outfilename,"wb"))
      
    # reset for next iter
    x0 = np.copy(xopt)
    if copt >=ctol:
      # only increase penalty if infeasible
      pen_param = pen_inc*pen_param
    else:
      # check stationarity
      if KKT:
        print("KKT conditions satisfied")
        break
      elif fopt <= ftarget:
        # decrease target to get more iterations
        ftarget = ftarget/10
  
