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
Check if we have a branch change.
Run with simsopt 0.12.2

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
    qs0 = qsrr.residuals()


    # compute a gradient with MPI
    with MPIFiniteDifference(qsrr.residuals, mpi, abs_step=1e-7,
                             rel_step=1e-5) as fd:
        if mpi.proc0_world:
            # jacobian J or residauls r
            qsrr_jac = fd.jac()
            # grad = J.T @ r0
            qsrr_grad = qsrr_jac.T @ qs0
            # H = J.T @ J
            H = qsrr_jac.T @ qsrr_jac
            Q = np.linalg.cholesky(H)
            # H @ r = gradf - H @ x0
            #r = np.linalg.solve(Q.T, np.linalg.solve(Q,qsrr_grad)) - x0
            qsrr_standard_grad = np.linalg.solve(Q,qsrr_grad)
    with MPIFiniteDifference(vmec.aspect, mpi, abs_step=1e-7,
                             rel_step=1e-5) as fd:
        if mpi.proc0_world:
            aspect_grad = fd.jac().flatten()

    if mpi.proc0_world:
        print(qsrr_jac.shape,aspect_grad.shape)
        print('norm qs grad', np.linalg.norm(qsrr_grad))
        print('norm qs transformed grad', np.linalg.norm(qsrr_standard_grad))
        print('norm aspect grad', np.linalg.norm(aspect_grad))
        print('dot product', aspect_grad @ qsrr_grad)
        lam = -aspect_grad @ qsrr_grad / (aspect_grad @ aspect_grad)
        print('lagrange multiplier',lam)
        rho = aspect_grad @ qsrr_grad/ np.linalg.norm(qsrr_grad)/np.linalg.norm(aspect_grad)
        print('correlation',rho)
        print('angle',np.arccos(rho))
