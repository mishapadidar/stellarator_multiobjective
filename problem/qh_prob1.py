import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from mpi4py import MPI
import sys
sys.path.append("../utils")
from divide_work import divide_work


class QHProb1():

  """
  Quasi-Helical problem with squared objectives
    F = [(QS - QS_target)**2,(aspect-aspect_target)**2]
  """
    
  def __init__(self,vmec_input="input.nfp4_QH_warm_start_high_res",n_partitions=0,max_mode=2):
    """
    n_partitions: number of MPI partitions to create. 
        Using multiple partitions is useful if you are implementing concurrent function
        evaluations in your driver script, or if you would like to distribute
        finite difference computations over multiple groups.
        Defaults to MPI.size()
    max_mode: maximum Fourier mode for the description of the input variables.
    """

    # load vmec and mpi
    if n_partitions == 0:
      n_partitions = MPI.COMM_WORLD.Get_size()
    self.n_partitions = n_partitions
    self.mpi = MpiPartition(n_partitions)
    self.vmec = Vmec(vmec_input, mpi=self.mpi,keep_all_files=False)

    # set vmec resolution (should be higher than boundary resolution)
    #self.vmec.indata.mpol = max_mode + 5
    #self.vmec.indata.ntor = max_mode + 5

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
    self.n_qs_radii = 11 
    self.QS = QuasisymmetryRatioResidual(self.vmec,
                                np.linspace(0,1,self.n_qs_radii),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
    self.n_qs_residuals = 44352 # counted from computation
    self.aspect_target = 7.0
    self.dim_raw = self.n_qs_residuals + 1
    self.dim_F = 2

  def sync_seeds(self):
    """
    Sync the np.random.seed of the various worker groups.
    The seed is a random number <1e6.
    """
    seed = np.zeros(1)
    if self.mpi.proc0_world:
      seed = np.random.randint(int(1e6))*np.ones(1)
    self.mpi.comm_world.Bcast(seed,root=0)
    np.random.seed(int(seed[0]))
    return int(seed[0])

  def raw(self,y):
    """
    Return the raw simulation output
    y: input variables describing surface
    y: array of size (self.dim_x,)

    return: objectives [QS residuals, aspect ratio].
            QS total is the sum of QS residuals squared. The residuals
            may be negative. The aspect ratio is single number
            whereas there are self.n_qs_residuals QS residuals.
    return: array of size (self.dim_F,)
    """
    # update the surface
    self.surf.x = np.copy(y)

    # evaluate the objectives
    try: 
      obj1 = self.QS.residuals()
    except: # catch failures
      obj1 = np.inf*np.ones(self.n_qs_residuals)
    try:
      obj2 = self.vmec.aspect()
    except:
      obj2 = np.inf

    obj = np.hstack((obj1,obj2))
    # catch partial failures
    obj[np.isnan(obj)] = np.inf
    return obj

  def rawp(self,Y):
    """
    Evaluate the raw objective vector at many points by distributing
    the resources over worker groups.

    This function expects all workers groups to contribute, so 
    all worker groups must evaluate this function on the same
    set of points Y.

    Y: array of input variables describing surface
    Y: array of size (n_points, self.dim_x).

    return: array of objectives [QS, aspect]
    return: array of size (n_points, self.dim_F)
    """
    n_points = np.shape(Y)[0]

    # special case for 1 partition
    if self.n_partitions ==1:
      return np.array([self.raw(y) for y in Y])

    # divide the evals across groups
    idxs,counts = divide_work(n_points,self.n_partitions)
    idx = idxs[self.mpi.group]

    # do the evals
    f   = np.array([self.raw(y) for y in Y[idx]]).flatten()

    # head leader gathers all evals
    F = np.zeros(n_points*self.dim_F)
    counts = self.dim_F*np.array(counts).astype(int)
    self.mpi.comm_leaders.Gatherv(f,(F,counts),root=0)
    # broadcast to leaders
    self.mpi.comm_leaders.Bcast(F,root=0)
    # broadcast internally within group
    self.mpi.comm_groups.Bcast(F,root=0)

    # reshape to 2D-array
    F = np.reshape(F,(-1,self.dim_F))
    
    return np.copy(F)

  def eval(self,y,raw=None):
    """
    Evaluate the objective vector.
    y: input variables describing surface
    y: array of size (self.dim_x,)
    raw: (optional), precomputed raw simulation values
         from self.raw(y)

    return: objectives [QS objective, aspect objective]
    return: array of size (self.dim_F,)
    """
    if raw is None:
      raw = self.raw(y)

    obj1 = np.mean(raw[:-1]**2)
    obj2 = (raw[-1] - self.aspect_target)**2
    obj =  np.array([obj1,obj2])
    # catch partial failures
    obj[np.isnan(obj)] = np.inf
    return obj

  def evalp(self,Y,Raw=None):
    """
    Evaluate the objective vector at many points by distributing
    the resources over worker groups.

    This function expects all workers groups to contribute, so 
    all worker groups must evaluate this function on the same
    set of points Y.

    Y: array of input variables describing surface
    Y: array of size (n_points, self.dim_x).     
    Raw: (optional), array of precomputed raw simulation output
    Raw: array of size (n_points, self.dim_F).

    return: array of objectives [QS objective, aspect objective]
    return: array of size (n_points, self.dim_F)
    """
    n_points = np.shape(Y)[0]

    # do the evals in parallel
    if Raw is None:
      Raw = self.rawp(Y)
    # turn the raw into objectives
    F   = np.array([self.eval(Y[ii],Raw[ii]) for ii in range(n_points)])
    return np.copy(F)

  def jac(self,y,h=1e-7,method='forward'):
    """
    Evaluate the objective jacobian. Use this
    function if n_partitions=1 or if distinct
    worker groups are evaluating jacobians at distinct
    points. 

    y: input variables describing surface
    y: array of size (self.dim_x,)
    h: step size
    h: float
    method: 'forward' or 'central'
    method: string

    return: jacobian of self.eval
    return: array of size (self.dim_F,self.dim_x)
    """
    assert method in ['forward','central']

    h2   = h/2.0
    Ep   = y + h2*np.eye(self.dim_x)
    Fp   = np.array([self.eval(e) for e in Ep])
    if method == 'forward':
      jac = (Fp - self.eval(y))/(h2)
    elif method == 'central': # central difference
      Em   = y - h2*np.eye(self.dim_x)
      Fm   = np.array([self.eval(e) for e in Em])
      jac = (Fp - Fm)/(h)
    return np.copy(jac.T)

  def jacp(self,y,h=1e-7,method='forward'):
    """
    Evaluate the objective jacobian by distributing the 
    finite difference over the worker groups. 
    Use this function if n_partitions > 1 AND
    if all worker groups are evaluating the jacobian 
    together at the same point, such as in a single
    optimization run. Do not use this function for 
    concurrent jacobian evaluations at distinct points.

    y: input variables describing surface
    y: array of size (self.dim_x,)
    h: step size
    h: float
    method: 'forward' or 'central'
    method: string

    return: jacobian of self.eval
    return: array of size (self.dim_F,self.dim_x)
    """
    assert method in ['forward','central']

    # special case
    if self.n_partitions == 1:
      return self.jac(y,h,method)

    h2   = h/2.0
    Ep   = y + h2*np.eye(self.dim_x)
    Fp   = self.evalp(Ep)
    if method == 'forward':
      jac = (Fp - self.eval(y))/(h2)
    elif method == 'central': # central difference
      Em  = y - h2*np.eye(self.dim_x)
      Fm  = self.evalp(Em)
      jac = (Fp - Fm)/(h)
    
    return np.copy(jac.T)

if __name__=="__main__":

  # test 1:
  # evaluate obj and jac with one partition
  test_1 = False
  if test_1 == True:
    prob = QHProb1(n_partitions=1,max_mode=2)
    x0 = prob.x0 
    import time
    t0 = time.time()
    raw = prob.raw(x0)
    print(raw)
    print(prob.eval(x0,raw))
    print(prob.eval(x0))
    print("\n\n\n")
    print('eval time',time.time() - t0)
    t0 = time.time()
    jac = prob.jac(x0,method='forward',h=1e-7)
    print("\n\n\n")
    print('jac time',time.time() - t0)
    print(jac)
  
  # test 2:
  # evaluate obj and jac with multiple partition
  test_2 = True
  if test_2 == True:
    prob = QHProb1(max_mode=1)
    x0 = prob.x0
    n_evals = 10
    Y = x0 + 1e-5*np.random.randn(n_evals,prob.dim_x)
    import time
    t0 = time.time()
    Raw = prob.rawp(Y)
    print(Raw)
    print(prob.evalp(Y,Raw))
    print(prob.evalp(Y))
    print("\n\n\n")
    print('eval time',time.time() - t0)
    quit()

    t0 = time.time()
    jac = prob.jacp(x0,method='forward',h=1e-7)
    print("\n\n\n")
    print('jac time',time.time() - t0)
    print(prob.mpi.group,jac)

