import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference

class QHProb1():

  """
  Quasi-Helical problem with squared objectives
    F = [(QS - QS_target)**2,(aspect-aspect_target)**2]
  """
    
  def __init__(self,vmec_input="input.nfp4_QH_warm_start",n_partitions=1,max_mode=2):
    # load vmec and mpi
    self.n_partitions = n_partitions
    self.mpi = MpiPartition(n_partitions)
    self.vmec = Vmec(vmec_input, mpi=self.mpi,keep_all_files=False)

    # set vmec resolution (should be higher than boundary resolution)
    self.vmec.indata.mpol = max_mode + 3
    self.vmec.indata.ntor = max_mode + 3

    # define parameters
    surf = self.vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # fix the Major radius

    # variables
    self.surf = surf # our variables
    self.x0 = self.surf.x # nominal starting point
    self.dim_x = len(surf.x) # dimension

    # objectives
    self.dim_F = 2
    self.QS = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
    self.qs_target = 0.0
    self.aspect_target = 7.0

  def eval(self,y):
    """
    Evaluate the objective vector.
    y: input variables describing surface
    y: array of size (self.dim_x,)

    return: objectives [QS objective, aspect objective]
    return: array of size (self.dim_F,)
    """
    # update the surface
    self.surf.x = np.copy(y)
    # evaluate the objectives
    try: 
      obj1 = (self.QS.total() - self.qs_target)**2
      obj2 = (self.vmec.aspect() - self.aspect_target)**2
    except: # catch failures
      obj1 = np.inf
      obj2 = np.inf
    return np.array([obj1,obj2])

  def jac(self,y):
    """
    WARNING: Method is UNTESTED
    Evaluate the objective jacobian.
    y: input variables describing surface
    y: array of size (self.dim_x,)

    return: objectives [QS objective, aspect objective]
    return: array of size (self.dim_F,)
    """
    # update the surface
    self.surf.x = np.copy(y)
    # evaluate the objectives
    try: 
      with MPIFiniteDifference(self.QS.total, self.mpi, diff_method="forward", abs_step=1e-7) as fd:
        if self.mpi.proc0_world:
            grad_qs = fd.jac()
      with MPIFiniteDifference(self.vmec.aspect, self.mpi, diff_method="forward", abs_step=1e-7) as fd:
        if self.mpi.proc0_world:
            grad_aspect = fd.jac()
      jac1 = 2*(self.QS.total() - self.qs_target)*grad_qs
      jac2 = 2*(self.vmec.aspect() - self.aspect_target)*grad_aspect
    except: # catch failures
      jac1 = np.inf*np.ones(self.dim_x)
      jac2 = np.inf*np.ones(self.dim_x)
    return np.array([jac1,jac2])

if __name__=="__main__":
  # define a multiobjective problem
  prob = QHProb1()
  x0 = prob.x0
  print(prob.eval(x0))
  print(prob.jac(x0))
  
