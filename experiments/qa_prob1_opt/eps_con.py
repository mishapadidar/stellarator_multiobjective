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
import qa_prob1
from eval_wrapper import eval_wrapper
#from gradient_descent import GD
from gauss_newton import GaussNewton
from block_coordinate_gauss_newton import BlockCoordinateGaussNewton

"""
epsilon contraint method for minimizing [Q(x),A(x)]
where Q is the mean squared quasisymmetry error
and A is the aspect ratio.

We solve
     min Q(x)
s.t. A(x) == A*

with a penalty approach
  p(x) = Q(x) + pen*max(0,A(x)-A*)**2
"""

#####
## take inputs
#####

# load the aspect ratio target
aspect_target = float(sys.argv[1])  # positive float
warm_start = sys.argv[2]=="True" # bool
vmec_res = sys.argv[3] # vmec input fidelity low, mid, high
max_mode = int(sys.argv[4]) # max mode = 1,2,3,4,5...

assert max_mode <=5, "max mode out of range"
#assert vmec_res in ["low","high"]
assert vmec_res in ["cold","low","high"]
if vmec_res == "cold":
  vmec_input = "../../../problem/input.nfp2_QA_cold"
elif vmec_res == "low":
  vmec_input = "../../../problem/input.nfp2_QA"
  if debug:
    vmec_input = "../../problem/input.nfp2_QA"
elif vmec_res == "high":
  vmec_input = "../../../problem/input.nfp2_QA_high_res"
  if debug:
    vmec_input = "../../problem/input.nfp2_QA_high_res"
#elif vmec_res == "super":
#  vmec_input = "../../../problem/input.nfp4_QH_warm_start_super_high_res"

# load the problem
prob = qa_prob1.QAProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)

# output directory
outputdir = "../data"
if debug:
  outputdir = "./data"

if warm_start is True:
  # find a point on the pareto front
  #pareto_data = pickle.load(open(outputdir + "/pareto_optimal_points.pickle","rb"))
  pareto_data = pickle.load(open(outputdir + "/warm_start_points.pickle","rb"))
  idx_start = np.argmin(np.abs(pareto_data['FX'][:,0] - aspect_target))
  x0 = np.copy(pareto_data['X'][idx_start])
  del pareto_data
  # convert to higher dimension representation
  x0 = prob.increase_dimension(x0,max_mode)
  # reset just to be sure
  prob = qa_prob1.QAProb1(max_mode=max_mode,vmec_input = vmec_input,aspect_target = aspect_target)
else:
  x0 = prob.x0
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

#####
## initialize parameters and tolerances
#####

F0 = prob.eval(x0)
aspect0 = prob.aspect(x0)
iota0 = prob.iota(x0)
qs_mse0 = F0[0]
if master:
  print(f"Starting with qs mse {qs_mse0} and aspect {aspect0} and mean iota {iota0}.")

max_iter = 100 # evals per iteration
ftarget  = 1e-11
ftol_abs = ftarget*1e-5
kkt_tol  = 1e-7 
max_solves = 3 # number of penalty updates
pen0    = 1e3 # initial penalty param.
pen_inc = 10.0 # increase parameter
ctol    = 1e-6 # target constraint tolerance

# pen parameter initialization
raw0 = prob.raw(x0)
jac0 = prob.jacp_residuals(x0)
grad_qs = (2/prob.n_qs_residuals)*jac0[:-2].T @ raw0[:-2]
grad_asp = jac0[-2]
grad_iota = jac0[-1]
# increase the penalty param
pen_param = pen0*np.linalg.norm(grad_qs)/np.array([np.linalg.norm(grad_asp),np.linalg.norm(grad_iota)])

if master:
  print('')
  print('aspect target: ', aspect_target)
  print('initial penalty parameter: ',pen_param)
  print('initial |grad|: ',np.linalg.norm(grad_qs + pen_param[0]*grad_asp + pen_param[1]*grad_iota))
  print('initial |qs grad|: ',np.linalg.norm(grad_qs))
  print('initial |aspect grad|: ',np.linalg.norm(grad_asp))
  print('initial |iota grad|: ',np.linalg.norm(grad_iota))
  print('ftarget:',ftarget)
  print('ftol_abs:',ftol_abs)
  print('kkt tol:',kkt_tol)

#####
## define functions
#####

# wrap the raw objective
func_wrap = eval_wrapper(prob.raw,prob.dim_x,prob.dim_raw)

def Constraint(xx):
  """ constraint c(x) <= 0"""
  ev = prob.aspect(xx) - aspect_target
  return ev
# write the objective
def PenaltyObjective(xx):
  """ penalty obj 
  p(x) = Q(x) + pen*max(0,(A(x) - A*))**2  + pen*(iota - iota*)**2 
  """
  qs_mse = np.mean(prob.qs_residuals(xx)**2)
  iota_res = prob.iota(xx) - prob.iota_target
  asp_res = prob.aspect(xx) - aspect_target
  #ret = qs_mse + pen_param[0]*max(asp_res,0)**2 + pen_param[1]*(iota_res)**2
  ret = qs_mse + pen_param[0]*(asp_res)**2 + pen_param[1]*(iota_res)**2
  if master:
    print(f'f(x): {ret}, qs mse: {qs_mse}, asp_res: {asp_res}, iota_res: {iota_res}')
  return ret
# write the objective
def PenaltyResiduals(xx):
  """ weighted penalty residuals 
    [(1/sqrt(m))q1(x),...,(1/sqrt(m))qm(x),sqrt(pen)*max(A(x) - A*,0)]
  """
  # compute raw values
  raw = func_wrap(xx)
  # for print
  qs_mse = np.mean(raw[:-2]**2)
  asp = raw[-2] 
  iota = raw[-1]
  # compute residual
  #raw[-2] = max(asp - aspect_target,0)
  raw[-2] = asp - aspect_target
  raw[-1] = iota - prob.iota_target
  # weight the residuals
  raw[:-2] *= np.sqrt(1.0/prob.n_qs_residuals)
  raw[-2]  *= np.sqrt(pen_param[0])
  raw[-1]  *= np.sqrt(pen_param[1])
  ff = np.sum(raw**2)
  if master:
    print(f'f(x): {ff}, qs mse: {qs_mse}, aspect: {asp}, iota: {iota}')
  return raw
def JacPenaltyResiduals(xx,idx=range(dim_x),h=1e-7):
  """Weighted penalty residuals jacobian 
     Method is comptabile with 
     BlockCoordinateGaussNewton

     Jacobian of 
     [(1/sqrt(m))q1(x),...,(1/sqrt(m))qm(x),sqrt(pen)*max(A(x) - A*,0),sqrt(pen)*(iota-iota*)]
  """
  #jac = prob.jacp_residuals(xx)
  h2   = h/2.0
  Ep   = xx + h2*np.eye(dim_x)[idx]
  Ep   = np.vstack((Ep,xx))
  Fp   = prob.residualsp(Ep)
  jac = (Fp[:-1] - Fp[-1])/(h2)
  jac = np.copy(jac.T)
  # save the evals
  #Fp[:,-2] += aspect_target
  #Fp[:,-1] += prob.iota_target
  # dont save the center point b/c it has already been saved
  #func_wrap.X = np.append(func_wrap.X,Ep[:-1],axis=0)
  #func_wrap.FX = np.append(func_wrap.FX,Fp[:-1],axis=0)

  ## make sure to take gradient of max
  #asp = prob.aspect(xx)
  #if asp -aspect_target <= 0.0:
  #  jac[-2] = 0.0

  # weight the residuals
  jac[:-2] *= np.sqrt(1.0/prob.n_qs_residuals)
  jac[-2] *= np.sqrt(pen_param[0])
  jac[-1] *= np.sqrt(pen_param[1])
  if master:
    print('computing jac')
  return jac

#####
## run penalty method
#####

if master:
  print("Running penalty method")
  print(f"with {max_iter} steps per iteration")
  sys.stdout.flush()

for ii in range(max_solves):
  if master:
    print("\n")
    print("iteration",ii)
    print("pen_param",pen_param)
  xopt = GaussNewton(PenaltyResiduals,JacPenaltyResiduals,x0,max_iter=max_iter,ftarget=ftarget,ftol_abs=ftol_abs,gtol=1e-10)
  fopt = PenaltyObjective(xopt)
  copt = Constraint(xopt)
  if master:
    print("")
    #print(res)
    print('optimal obj:',fopt)
    print('optimal con:',copt)
    print('pen param:',pen_param)
    sys.stdout.flush()

  # compute gradients at minimum
  rawopt = prob.raw(xopt)
  jacopt = prob.jacp_residuals(xopt)
  grad_qs = (2/prob.n_qs_residuals)*jacopt[:-2].T @ rawopt[:-2]
  grad_asp = jacopt[-2]
  grad_iota = jacopt[-1]

  # compute some values
  qs_mse_opt = np.mean(rawopt[:-2]**2)
  aspect_opt = rawopt[-2]
  iota_opt = rawopt[-1]
  
  if master:
    print('qs mse',qs_mse_opt)
    print('aspect',aspect_opt)
    print('iota',iota_opt)
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
    outdata['iota_opt'] = iota_opt
    outdata['copt'] = copt
    outdata['aspect_target'] = aspect_target
    outdata['iota_target'] = prob.iota_target
    outdata['ctol'] = ctol
    outdata['jacopt'] = jacopt 
    outdata['X'] = func_wrap.X
    outdata['RawX'] = func_wrap.FX
    outdata['pen_param'] = pen_param
    outdata['kkt_tol'] = kkt_tol
    outdata['grad_qs'] = grad_qs
    outdata['grad_aspect'] = grad_asp
    pickle.dump(outdata,open(outfilename,"wb"))
    
  # reset for next iter
  x0 = np.copy(xopt)
  #if copt >=ctol:
  if np.abs(aspect_opt-aspect_target) >=ctol:
    # only increase penalty if infeasible
    pen_param[0] = pen_inc*pen_param[0]
  if np.abs(iota_opt-prob.iota_target) >=ctol:
    # only increase penalty if infeasible
    pen_param[1] = pen_inc*pen_param[1]
  if fopt <= ftarget:
    # decrease target to get more iterations
    ftarget = ftarget/10

