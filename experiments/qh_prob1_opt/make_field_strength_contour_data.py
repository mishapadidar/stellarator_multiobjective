import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
import pickle
import sys

"""
Computes data for a contour plot of the field strength in boozer coordinates.

usage:
  mpiexec -n 1 python3 make_field_strength_contour_data.py
"""
infile_list = ["./data/input.nfp4_QH_aspect_3.5",
              "./data/input.nfp4_QH_aspect_5.5",
              "./data/input.nfp4_QH_aspect_8.7"]


for ii,vmec_input in enumerate(infile_list):
    # interpolation params
    interpolant_degree = 3
    interpolant_level = 8
    bri_mpol = 32
    bri_ntor = 32
    
    # make vmec
    mpi = MpiPartition()
    vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
    
    # Construct radial interpolant of magnetic field
    bri = BoozerRadialInterpolant(vmec, order=interpolant_degree,
                      mpol=bri_mpol,ntor=bri_ntor, enforce_vacuum=True)
    
    # Construct 3D interpolation
    nfp = vmec.wout.nfp
    srange = (0, 1, interpolant_level)
    thetarange = (0, np.pi, interpolant_level)
    zetarange = (0, 2*np.pi/nfp,interpolant_level)
    field = InterpolatedBoozerField(bri, degree=interpolant_degree, srange=srange, thetarange=thetarange,
                       zetarange=zetarange, extrapolate=True, nfp=nfp, stellsym=True)
    
    s_list = [0.05,0.25,0.5,1.0]
    ntheta = 128
    nzeta = 128
    thetas = np.linspace(0, 2*np.pi, ntheta)
    zetas = np.linspace(0,2*np.pi/nfp, nzeta)
    [thetas,zetas] = np.meshgrid(thetas, zetas)
    # storage
    modB_list = np.zeros((len(s_list),ntheta*nzeta))
    for ii,s_label in enumerate(s_list):
      # get a list of points
      stz_inits = np.zeros((ntheta*nzeta, 3))
      stz_inits[:, 0] = s_label
      stz_inits[:, 1] = thetas.flatten()
      stz_inits[:, 2] = zetas.flatten()
      # evaluate the field on the mesh
      field.set_points(stz_inits)
      modB = field.modB().flatten()
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
    filename = vmec_input.split("/")[-1] # remove any directories
    filename = filename[6:] # remove the "input."
    filename = filename + "_field_strength.pickle"
    filename = "./data/" + filename
    pickle.dump(outdata,open(filename,"wb"))
    
