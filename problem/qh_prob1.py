import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

class QHProb1():

  """
  Quasi-Helical problem with squared objectives
    F = [(QS - QS_target)**2,(aspect-aspect_target)**2]
  """
    
  def __init__(self,vmec_input="input.nfp4_QH_warm_start",n_partitions,max_mode=2):
    # load vmec and mpi
    mpi = MpiPartition(n_partitions)
    vmec = Vmec(vmec_input, mpi=mpi)

    # define parameters
    surf = vmec.boundary
    surf.fix_all()
    max_mode = max_mode
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # fix the Major radius

    # attributes
    self.surf = surf # our variables
    self.dim = len(surf.x) # dimension
    self.QS = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
    self.qs_target = 0.0
    self.aspect_target = 7.0

  def eval(self,y):
    # update the surface
    self.surf.x = np.copy(y)
    # evaluate the objectives
    try: 
      obj1 = (self.QS.total() - self.qs_target)**2
      obj2 = (vmec.aspect() - self.aspect_target)**2
    except: # catch failures
      obj1 = np.inf
      obj2 = np.inf
    return np.array([obj1,obj2])

if __name__=="__main__":
  # define a multiobjective problem
  prob = QHProb(surf,vmec)
  x0 = np.copy(surf.x)
  print(prob.eval(x0))
  
