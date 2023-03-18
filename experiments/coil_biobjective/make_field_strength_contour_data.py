import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
import pickle
import sys
import os
from pathlib import Path
import numpy as np
import pickle
import uuid
from scipy.optimize import minimize
import sys
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk, MeanSquaredCurvature, CurveSurfaceDistance
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty


"""
Use simsopt 0.12.2

Computes data for a contour plot of the field strength in boozer coordinates.

usage:
  mpiexec -n 1 python3 make_field_strength_contour_data.py
"""

# paraview files
infile_list = ["./output/biobjective/length/biobjective_eps_con_length_15.5_cold_ncoils_4_785bbf81-f8fb-488c-b0ef-12a2bbec651d.pickle",
    "./output/biobjective/length/biobjective_eps_con_length_19.555555555555557_warm_ncoils_4_0f97f07b-8d84-428f-925a-43bebaa5e441.pickle"]


# interpolation params
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 32
bri_ntor = 32
vmec_input = "../../vmec_input_files/input.LandremanPaul2021_QA_faster"

# make vmec
mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary

# coil params; same as used to generate them
R0 = surf.get("rc(0,0)")
R1 = R0/2
ncoils = 4
order = 5 # num fourier modes per coils
current = 1e5


nfp = surf.nfp
# (s,theta,zeta) to evaluate field at
s_list = [0.05,0.25,0.5,1.0]
ntheta = 128
nzeta = 128
thetas = np.linspace(0, 2*np.pi, ntheta)
zetas = np.linspace(0,2*np.pi/nfp, nzeta)
[thetas,zetas] = np.meshgrid(thetas, zetas)

# Construct radial interpolant of magnetic field
bri = BoozerRadialInterpolant(vmec, order=interpolant_degree,
                  mpol=bri_mpol,ntor=bri_ntor, enforce_vacuum=True)

for ii,infile in enumerate(infile_list):
    # load the point
    indata = pickle.load(open(infile,"rb"))
    xopt = indata['xopt']

    # load the coils
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
    JF.x = xopt

    # storage
    modB_list = np.zeros((len(s_list),ntheta*nzeta))
    for ii,s_label in enumerate(s_list):
      # get a list of points
      stz_inits = np.zeros((ntheta*nzeta, 3))
      stz_inits[:, 0] = s_label
      stz_inits[:, 1] = thetas.flatten()
      stz_inits[:, 2] = zetas.flatten()
      # map points to cylindrical coords
      bri.set_points(stz_inits)
      R = bri.R_ref().flatten()
      # phi = zeta - nu
      phi = zetas.flatten() - bri.nu_ref().flatten()
      Z = bri.Z_ref().flatten()
      rphiz_inits = np.zeros((ntheta*nzeta, 3))
      rphiz_inits[:, 0] = R
      rphiz_inits[:, 1] = phi
      rphiz_inits[:, 2] = Z
      # evaluate the field on the mesh
      bs.set_points_cyl(rphiz_inits)
      modB = bs.AbsB().flatten()
      # append to modB_list
      modB_list[ii] = np.copy(modB)
    
      # to reshape modB into mesh 
      #modB_mesh = np.reshape(modB,((nzeta,ntheta)))
    
    # dump the data
    outdata = {}
    outdata['s_list'] = s_list
    outdata['ntheta'] = ntheta
    outdata['nzeta'] = nzeta
    outdata['theta_mesh'] = thetas
    outdata['zeta_mesh'] = zetas
    outdata['modB_list'] = modB_list
    filename = infile.split("/")[-1] # remove any directories
    filename = filename[:-7] # remove the ".pickle"
    filename = filename + "_field_strength.pickle"
    filename = "./output/" + filename
    pickle.dump(outdata,open(filename,"wb"))
    
