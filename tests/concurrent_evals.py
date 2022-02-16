import time
import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

# two worker groups
n_partitions = 2
mpi = MpiPartition(n_partitions)

vmec = Vmec("../problem/input.nfp4_QH_warm_start", mpi=mpi)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)") # Major radius

print('Parameter space:', surf.dof_names)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# generate some points to evaluate
x0 = np.copy(surf.x)
dim = len(surf.x)
n_points = 10
# different groups should have different random seeds
X = x0 + 1e-4*np.random.randn((n_points,dim))
F = np.zeros(n_points)

t0 = time.time()
for ii,xx in enumerate(X1):
  # evaluate the objective
  surf.x = np.copy(xx)
  F[ii] = qs.total()
print('group: ',mpi.group, ', time:',time.time()-t0)
