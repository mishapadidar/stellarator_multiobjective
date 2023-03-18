#import os
import numpy as np
#from pathlib import Path
import glob
import pickle
#import uuid
#from scipy.optimize import minimize
#import sys
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk, MeanSquaredCurvature, CurveSurfaceDistance
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import QfmSurface
from simsopt.geo import QfmResidual, Volume

"""
Run with simsopt 0.12.2.
  mpiexec -n 1 python3 qfm_surface.py

Build QFM surfaces from the coil B-fields, solve the ideal MHD
equilibrium using the QFM surface as the boundary shape, 
compute the quasi-symmetry metric.
"""

# Initialize the boundary magnetic surface:
vmec_input = '../../vmec_input_files/input.LandremanPaul2021_QA_faster'
ntheta=nphi=32
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="half period", nphi=nphi, ntheta=ntheta)

# coil params
R0 = surf.get("rc(0,0)")
R1 = R0/2
ncoils = 4
order = 5 # num fourier modes per coils
current = 1e5

# quasi-symmetry params
helicity_m = 1
helicity_n = 0

# process the pareto optimal points
filelist = glob.glob("./output/pareto_data/*_ncoils_4_*.pickle")
n_configs = len(filelist)

# files we plotted in paraview
paraview_files = ["./output/pareto_data/biobjective_eps_con_length_14.0_cold_ncoils_4_45d6612c-1c84-4c56-b5cc-69aba4906332.pickle",
    "./output/pareto_data/biobjective_eps_con_length_19.333333333333332_warm_ncoils_4_139bd8e6-79ad-4c3b-84b2-216ba1449ee8.pickle"]
paraview_indexes = [ii for ii,xx in enumerate(filelist) if xx in paraview_files]
print(paraview_indexes)

# storage
outfilename = "./output/qfm_data.pickle"
outdata = {}
outdata['filelist'] = filelist
outdata['helicity_m'] = helicity_m
outdata['helicity_n'] = helicity_n
outdata['paraview_indexes'] = paraview_indexes

# more storage
qsrr_list = np.zeros(n_configs)
Fopt_list = np.zeros((n_configs,2))
ncoils_list = np.zeros(n_configs)

# measure quasi-symmetry of the surface
mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
vmec.boundary = surf
qsrr = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
print(f"Baseline Quasi-symmetry",qsrr.total())
outdata['surface_qsrr'] = qsrr.total()

for ifile, infile in enumerate(filelist):
    print("")
    print(infile)

    # load the point
    indata = pickle.load(open(infile,"rb"))
    xopt = indata['xopt']

    base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Jflux = SquaredFlux(surf, bs)
    Jlength = [CurveLength(c) for c in base_curves]
    Jlength_total = sum(Jlength)
    coil_surf_dist_rhs = 0.01
    Jcoil_surf_dist = CurveSurfaceDistance(base_curves,surf,coil_surf_dist_rhs)
    JF = Jflux + QuadraticPenalty(Jlength_total,0.0, f='max')\
        + Jcoil_surf_dist
    
    # target volume
    vol_target = surf.volume()
    
    # volume function
    vol = Volume(surf)
    
    JF.x = xopt

    qfmres = QfmResidual(surf, bs) # residual
    qfm_surface = QfmSurface(bs, surf, vol, vol_target)
    #res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1500,constraint_weight=1e4)
    print(res)
    print(f"||vol constraint||={0.5*(surf.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfmres.J()):.8e}")

    # vmec object for qs computations
    mpi = MpiPartition()
    vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
    vmec.boundary = surf
    qsrr = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
    
    print(f"Quasi-symmetry",qsrr.total())


    # save important values
    qsrr_list[ifile] = qsrr.total()
    Fopt_list[ifile] = indata['Fopt']
    ncoils_list[ifile] = indata['ncoils']

    # dump data
    outdata['qsrr_list'] = qsrr_list
    outdata['Fopt_list'] = Fopt_list
    outdata['ncoils_list'] = ncoils_list
    pickle.dump(outdata,open(outfilename,"wb"))
