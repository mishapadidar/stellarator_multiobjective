import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from mpi4py import MPI
import sys
sys.path.append("../utils")
from divide_work import divide_work


class QAProb1():

  """
  Quasi-Axisymmetric problem with squared objectives
    F = [(QS - QS_target)**2,(aspect-aspect_target)**2]
  """
    
  def __init__(self,vmec_input="input.nfp2_QA",n_partitions=0,
    max_mode=2,aspect_target=6.0):
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
    self.vmec = Vmec(vmec_input, mpi=self.mpi,keep_all_files=False,verbose=False)

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
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
    self.n_qs_residuals = 44352 # counted from computation
    self.aspect_target = aspect_target
    self.dim_raw = self.n_qs_residuals + 1
    self.dim_F = 2

  def increase_dimension(self,xx,target_max_mode):
    """
    Convert a vector xx to a higher mode representation.
    """
    dims = [8,24,48,80,120] # modes 1,2,3,4,5
    dim_x = len(xx)
    current_max_mode = dims.index(dim_x)+1
    assert current_max_mode <= target_max_mode,"Need larger mode target"
    # create a surface object with the right max_mode
    surf = self.vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=current_max_mode,
                     nmin=-current_max_mode, nmax=current_max_mode, fixed=False)
    surf.fix("rc(0,0)") # fix the Major radius
    # set x
    surf.x = xx
    # now increase the mode
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=target_max_mode,
                     nmin=-target_max_mode, nmax=target_max_mode, fixed=False)
    surf.fix("rc(0,0)") # Major radius
    return surf.x

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

  def aspect(self,y):
    """ compute aspect ratio at a single point"""
    # update the surface
    self.surf.x = np.copy(y)

    # evaluate the objectives
    try:
      obj = self.surf.aspect_ratio()
    except:
      obj = np.inf

    # catch partial failures
    if np.isnan(obj):
      obj = np.inf
    return obj

  def qs_residuals(self,y):
    """ compute qs residuals at a single point"""
    # update the surface
    self.surf.x = np.copy(y)

    # evaluate the objectives
    try: 
      obj = self.QS.residuals()
    except: # catch failures
      obj = np.inf*np.ones(self.n_qs_residuals)

    # catch partial failures
    obj[np.isnan(obj)] = np.inf
    return obj
   

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
      #obj2 = self.vmec.aspect()
      obj2 = self.surf.aspect_ratio()
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

    WARNING: This is memory intensive because dim_raw is quite large.

    Y: array of input variables describing surface
    Y: array of size (n_points, self.dim_x).

    return: array of objectives [QS residuals, aspect]
    return: array of size (n_points, self.dim_raw)
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
    F = np.zeros(n_points*self.dim_raw)
    counts = self.dim_raw*np.array(counts).astype(int)
    self.mpi.comm_leaders.Gatherv(f,(F,counts),root=0)
    # broadcast to leaders
    self.mpi.comm_leaders.Bcast(F,root=0)
    # broadcast internally within group
    self.mpi.comm_groups.Bcast(F,root=0)

    # reshape to 2D-array
    F = np.reshape(F,(-1,self.dim_raw))
    
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
      """
      Do the evals directly from .eval() rather than .rawp()
      to avoid the memory burden of message passing the output
      of .raw()
      """
      # special case for 1 partition
      if self.n_partitions ==1:
        return np.array([self.eval(y) for y in Y])

      # divide the evals across groups
      idxs,counts = divide_work(n_points,self.n_partitions)
      idx = idxs[self.mpi.group]

      # do the evals
      f   = np.array([self.eval(y) for y in Y[idx]]).flatten()

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

    else:  
      # turn the raw into objectives
      F   = np.array([self.eval(Y[ii],Raw[ii]) for ii in range(n_points)])

    return np.copy(F)

  def residuals(self,y):
    """
    compute the residuals at y
    """
    # compute raw objectives
    resid = self.raw(y)
    # turn aspect into a residual
    resid[-1] -= self.aspect_target
    # catch partial failures
    resid[np.isnan(resid)] = np.inf
    return resid

  def residualsp(self,Y):
    """
    Evaluate the residual vector at many points by distributing
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
    # do the evals in parallel
    resid = self.rawp(Y)
    # turn aspect into a residual
    resid[:,-1] -= self.aspect_target
    # catch partial failures
    resid[np.isnan(resid)] = np.inf
    return resid

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

  def jac_residuals(self,y,h=1e-7,method='forward'):
    """
    Evaluate the residulas jacobian. Use this
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
    Fp   = np.array([self.residuals(e) for e in Ep])
    if method == 'forward':
      jac = (Fp - self.residuals(y))/(h2)
    elif method == 'central': # central difference
      Em   = y - h2*np.eye(self.dim_x)
      Fm   = np.array([self.residuals(e) for e in Em])
      jac = (Fp - Fm)/(h)
    return np.copy(jac.T)

  def jacp_residuals(self,y,h=1e-7,method='forward'):
    """
    Evaluate the residuals jacobian by distributing the 
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
      return self.jac_residuals(y,h,method)

    h2   = h/2.0
    Ep   = y + h2*np.eye(self.dim_x)
    Fp   = self.residualsp(Ep)
    if method == 'forward':
      jac = (Fp - self.residuals(y))/(h2)
    elif method == 'central': # central difference
      Em  = y - h2*np.eye(self.dim_x)
      Fm  = self.residualsp(Em)
      jac = (Fp - Fm)/(h)
    
    return np.copy(jac.T)

if __name__=="__main__":

  # test 1:
  # evaluate obj and jac with one partition
  test_1 = False
  if test_1 == True:
    prob = QAProb1(n_partitions=1,max_mode=1)
    x0 = prob.x0 
    import time
    t0 = time.time()
    raw = prob.raw(x0)
    print(raw)
    print(len(raw))
    print("\n\n\n")
    print('eval time',time.time() - t0)
    resid = prob.residuals(x0)
    print(resid)
    print(resid-raw)
    print(prob.eval(x0,raw))
    print(prob.eval(x0))
    t0 = time.time()
    jac = prob.jac(x0,method='forward',h=1e-7)
    print("\n\n\n")
    print('jac time',time.time() - t0)
    print(jac)
  
  # test 2:
  # evaluate obj and jac with multiple partition
  test_2 = True
  if test_2 == True:
    prob = QAProb1(vmec_input="input.nfp2_QA",max_mode=1)
    prob.sync_seeds()
    x0 = prob.x0
    n_evals = 10
    p = 1e-5*np.random.randn(n_evals,prob.dim_x)
    p[0] = 0.0
    Y = x0 + p
    import time
    t0 = time.time()
    Raw = prob.rawp(Y)
    print(Raw)
    Res = prob.residualsp(Y)
    print(Res)
    print(Raw - Res)
    print(prob.jacp_residuals(x0))
    quit()
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

