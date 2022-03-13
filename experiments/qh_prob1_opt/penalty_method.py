import numpy as np
import pickle
from datetime import datetime
from scipy.optimize import minimize
import sys
sys.path.append("../../../utils")
sys.path.append("../../../optim")
sys.path.append("../../../problem")
import qh_prob1
from eval_wrapper import eval_wrapper
from gradient_descent import GD
from gauss_newton import GaussNewton

"""
Penalty Method to solve
  min QS^2
  s.t. aspect = aspect_target
"""

# load the aspect ratio target
aspect_target = float(sys.argv[1]) 
outputdir = sys.argv[2] # should be formatted as i.e. "../data"
try:
  warm_start_file = sys.argv[3]
  warm_start = True
except:
  warm_start = False

# load the problem
#vmec_input = "../../../problem/input.nfp4_QH_warm_start_high_res"
vmec_input = "../../../problem/input.nfp4_QH_warm_start"
prob = qh_prob1.QHProb1(vmec_input = vmec_input,aspect_target = aspect_target)
if warm_start == False:
  x0 = prob.x0
else:
  x0 = pickle.load(open(warm_start_file,"rb"))['xopt']
dim_x = prob.dim_x

# mpi rank
if prob.mpi.proc0_world:
  master = True
else:
  master = False

# set outfile
now     = datetime.now()
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
outfilename = outputdir + f"/data_aspect_{aspect_target}_{barcode}.pickle"

# pen parameter initialiization
jac = prob.jacp(x0)
pen_param = np.linalg.norm(jac[0])/np.linalg.norm(jac[1])
#pen_param*=100
if master:
  print('')
  print('aspect target: ', aspect_target)
  print('initial penalty parameter: ',pen_param)
  print('initial |grad|: ',np.linalg.norm(jac[0] + pen_param*jac[1]))
  print('initial |qs grad|: ',np.linalg.norm(jac[0]))
  print('initial |aspect^2 grad|: ',np.linalg.norm(jac[1]))

# set relative gtol
gtol     = 1e-3*np.linalg.norm(jac[0] + pen_param*jac[1]) # stopping tolerance
qs_gtol  = 1e-4*np.linalg.norm(jac[0]) # target gtol for qs

if master:
  print('gtol:',gtol)
  print('qs gtol:',qs_gtol)

max_eval = 1000 # evals per iteration
max_solves = 7 # number of penalty updates
pen_inc = 10.0 # increase parameter
ctol    = 1e-6 # target constraint tolerance
method  ='L-BFGS-B'
options = {'maxfun':max_eval,'gtol':gtol}

def constraint(xx):
  """ constraint """
  prob.eval(xx)
  ev = prob.vmec.aspect() - aspect_target
  return ev
# write the objective
def objective(xx):
  """ penalty obj """
  ev  = prob.eval(xx)
  #ev  = func_wrap(xx)
  ret = ev[0] + pen_param*ev[1]
  if master:
    print(f'f(x): {ret}, qs err: {ev[0]}, aspect^2: {ev[1]}')
  return ret
def gradient(xx):
  """ penalty jac """
  jac = prob.jacp(xx)
  ret = jac[0] + pen_param*jac[1]
  if master:
    print('|grad|',np.linalg.norm(ret))
  return ret
# write the objective
def residuals(xx):
  """ penalty residuals """
  # compute residuals
  resid = prob.residuals(xx)
  qs_err = np.mean(resid[:-1]**2)
  asp_err = resid[-1] 
  # weight the residuals
  n_qs_resid = len(resid - 1)
  resid[:-1] *= np.sqrt(1.0/n_qs_resid)
  resid[-1] *= np.sqrt(pen_param)
  ff = np.sum(resid**2)
  if master:
    print(f'f(x): {ff}, qs mean square: {qs_err}, aspect res: {asp_err}')
  return resid
def jac_residuals(xx):
  """ penalty residuals jac """
  jac = prob.jacp_residuals(xx)
  # weight the residuals
  n_qs_resid = len(jac - 1)
  jac[:-1] *= np.sqrt(1.0/n_qs_resid)
  jac[-1] *= np.sqrt(pen_param)
  if master:
    print('computing jac')
  return jac

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
  #res     = minimize(obj,x0,jac=grad,method=method,options=options)
  #xopt = res.x
  #xopt = GD(objective,gradient,x0,alpha0 = 1e-1,gamma=0.5,max_iter=max_eval,gtol=gtol,c_1=1e-6,verbose=False)
  xopt = GaussNewton(residuals,jac_residuals,x0,max_iter=max_eval,gtol=gtol)
  fopt = objective(xopt)
  copt = constraint(xopt)
  if master:
    print("\n\n\n")
    #print(res)
    print('optimal obj:',fopt)
    print('optimal con:',copt)
    sys.stdout.flush()

  # reset for next iter
  x0 = np.copy(xopt)
  if np.abs(copt) >ctol:
    # only increase penalty if needed
    pen_param = pen_inc*pen_param
  else:
    # check stationarity
    grad_qs = prob.jacp(xopt)[0]
    if np.linalg.norm(grad_qs) <= qs_gtol:
      break
    else:
      gtol = qs_gtol/2

  # get run data
  #X = func_wrap.X
  #FX = func_wrap.FX
  
  # dump the evals at the end
  if master:
    print("\n\n\n")
    print(f"Dumping data to {outfilename}")
    outdata = {}
    outdata['xopt'] = xopt
    outdata['fopt'] = fopt
    outdata['copt'] = copt
    outdata['pen_param'] = pen_param
    #outdata['X'] = X
    #outdata['FX'] = FX
    outdata['aspect_target'] = aspect_target
    pickle.dump(outdata,open(outfilename,"wb"))
    
