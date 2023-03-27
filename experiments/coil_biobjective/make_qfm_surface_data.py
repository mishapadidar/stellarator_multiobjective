import os
import numpy as np
#from pathlib import Path
import glob
import pickle
#import uuid
#from scipy.optimize import minimize
import sys
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk, MeanSquaredCurvature, CurveSurfaceDistance
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import QfmSurface
from simsopt.geo import QfmResidual, Volume
debug = False

"""
Run with simsopt 0.12.2.

Run using the batch submission script
  python3 batch_submit_qfm.py
or
  mpiexec -n 1 python3 qfm_surface.py filename

Build QFM surfaces from the coil B-fields, solve the ideal MHD
equilibrium using the QFM surface as the boundary shape, 
compute the quasi-symmetry metric.
"""

# Initialize the boundary magnetic surface:
vmec_input = '../../../vmec_input_files/input.LandremanPaul2021_QA_faster'
if debug:
    vmec_input = vmec_input[3:]


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
infile = sys.argv[1]
if not debug:
    infile = "../" + infile[infile.find("/")+1:]
print("processing", infile)


## files we plotted in paraview
#paraview_files = ["./output/pareto_data/biobjective_eps_con_length_15.5_cold_ncoils_4_785bbf81-f8fb-488c-b0ef-12a2bbec651d.pickle",
#    "./output/pareto_data/biobjective_eps_con_length_19.555555555555557_warm_ncoils_4_0f97f07b-8d84-428f-925a-43bebaa5e441.pickle"]

# storage
outfilename = infile.split("/")[-1] # remove any directories
outfilename = outfilename[:-7] # remove the ".pickle"
outfilename = outfilename + "_qfm_data.pickle"
#outfilename = "./output/qfm_data/" + outfilename
outdir = "./"
outfilename = outdir + outfilename
#if not os.path.exists("./output/qfm_data"):
#    os.makedirs("./output/qfm_data")
outdata = {}
outdata['infile'] = infile
outdata['helicity_m'] = helicity_m
outdata['helicity_n'] = helicity_n


# measure quasi-symmetry of the surface
mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
vmec.boundary = surf
qsrr = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
print(f"Baseline Quasi-symmetry",qsrr.total())
outdata['surface_qsrr'] = qsrr.total()

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
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=2000,constraint_weight=1e4)
print(res)
print(f"||vol constraint||={0.5*(surf.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfmres.J()):.8e}")

# vmec object for qs computations
mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
vmec.boundary = surf
qsrr = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|

try:
    # vmec might fail
    print(f"Quasi-symmetry",qsrr.total())
except:
    print("\n")
    print("quasi-symmetry computation failed")
    quit()

# make a vtk file of the surface
surfname = infile[:-7] # remove the ".pickle"
surfname += "_qfm_surface"
surf.to_vtk(surfname)

# dump data
outdata['qsrr'] = qsrr.total()
outdata['Fopt'] = indata['Fopt']
outdata['ncoils'] = indata['ncoils']
print("Dumped", outfilename)
pickle.dump(outdata,open(outfilename,"wb"))
