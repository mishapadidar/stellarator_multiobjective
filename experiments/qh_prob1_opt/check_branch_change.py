import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.mhd import Vmec
import pickle
import sys


"""
Check if we have a branch change

usage:
  mpiexec -n 96 python3 check_branch_change.py
"""
# find efficient points with these aspect ratios
aspect_list = [6.5, 8.0]

# load the pareto set
infilename = "./data/pareto_optimal_points.pickle"
indata = pickle.load(open(infilename,"rb"))
X = indata['X']
FX = indata['FX']
aspect = np.copy(FX[:,0])
qs = np.copy(FX[:,1])

mpi = MpiPartition()

for aspect_target in aspect_list:

    # find the closest point to the target
    idx = np.argmin(np.abs(aspect - aspect_target))
    x0 = X[idx]
    if mpi.proc0_world:
        print("")
        print('qs_mse',qs[idx],'aspect',aspect[idx])
    
    # evaluate the point
    max_mode = 5
    vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
    vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # fix the Major radius

    surf.x = x0

    # define the QS objective
    qsrr = QuasisymmetryRatioResidual(vmec,
                                np.linspace(0,1,11),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

    # compute a gradient with MPI
    with MPIFiniteDifference(qsrr.total, mpi, abs_step=1e-7,
                             rel_step=1e-5) as fd:
        if mpi.proc0_world:
            qsrr_jac = fd.jac().flatten()
    with MPIFiniteDifference(vmec.aspect, mpi, abs_step=1e-7,
                             rel_step=1e-5) as fd:
        if mpi.proc0_world:
            aspect_jac = fd.jac().flatten()

    if mpi.proc0_world:
        print(qsrr_jac.shape,aspect_jac.shape)
        print('norm qs grad', np.linalg.norm(qsrr_jac))
        print('norm aspect grad', np.linalg.norm(aspect_jac))
        print('dot product', aspect_jac @ qsrr_jac)
        rho = aspect_jac @ qsrr_jac/ np.linalg.norm(qsrr_jac)/np.linalg.norm(aspect_jac)
        print('correlation',rho)
        print('angle',np.arccos(rho))
