
import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives.graph_least_squares import LeastSquaresProblem
from simsopt.solve.graph_mpi import least_squares_mpi_solve

mpi = MpiPartition(7)

vmec = Vmec("../problem/input.nfp4_QH_warm_start", mpi=mpi)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)") # Major radius

print("\n\n\n\n")
print('Parameter space:', surf.dof_names)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.residuals, 0, 1)])

print("\n\n\n\n")
print("Quasisymmetry objective before optimization:", qs.total())
print("Total objective before optimization:", prob.objective())

import time
t0 = time.time()

print("\n\n\n\n")
least_squares_mpi_solve(prob, mpi, grad=True)

print("\n\n\n\n")
print('Time to solve: ',time.time() - t0)
print('Total Obj: ',prob.objective())
print('Total QS: ',qs.total())
