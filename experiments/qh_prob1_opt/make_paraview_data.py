import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import pickle
import sys


"""
Make the data for paraview plotting

usage:
  mpiexec -n 1 python3 make_paraview_data.py
"""
infile_list = []

for ii,vmec_input in enumerate(infile_list):

    vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
    n_partitions=1
    mpi = MpiPartition(n_partitions)
    vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
    
    # generate the .wout data
    vmec.run()

    # load in the .wout data
    ns = vmec.wout.ns
    xn = vmec.wout.xn
    xm = vmec.wout.xm
    xn_nyq = vmec.wout.xn_nyq
    xm_nyq = vmec.wout.xm_nyq
    rmnc = vmec.wout.rmnc
    zmns = vmec.wout.zmns
    
    lmns = vmec.wout.lmns
    bmnc = vmec.wout.bmnc
    #raxis_cc = f.variables['raxis_cc'][()]
    #zaxis_cs = f.variables['zaxis_cs'][()]
    #buco = f.variables['buco'][()]
    #bvco = f.variables['bvco'][()]
    #jcuru = f.variables['jcuru'][()]
    #jcurv = f.variables['jcurv'][()]
    lasym = vmec.wout.lasym
    if lasym==1:
        rmns = vmec.wout.rmns
        zmnc = vmec.wout.zmnc
        lmnc = vmec.wout.lmnc
        bmns = vmec.wout.bmns
        #raxis_cs = vmec.wout.raxis_cs
        #zaxis_cc = vmec.wout.zaxis_cc
    else:
        rmns = 0*rmnc
        zmnc = 0*rmnc
        lmnc = 0*rmnc
        bmns = 0*bmnc
        #raxis_cs = 0*raxis_cc
        #zaxis_cc = 0*raxis_cc
    
    rmnc = rmnc.T
    rmns = rmns.T
    zmnc = zmnc.T
    zmns = zmns.T
    bmnc = bmnc.T
    bmns = bmns.T
    
    ntheta = 128
    nphi = 150
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    phi1D = np.linspace(0,2*np.pi,num=nphi)
    phi2D, theta2D = np.meshgrid(phi1D,theta1D)
    iradius = ns-1
    R = np.zeros((ntheta,nphi))
    Z = np.zeros((ntheta,nphi))
    B = np.zeros((ntheta,nphi))
    nmodes = len(xn)
    for imode in range(nmodes):
        angle = xm[imode]*theta2D - xn[imode]*phi2D
        R = R + rmnc[iradius,imode]*np.cos(angle) + rmns[iradius,imode]*np.sin(angle)
        Z = Z + zmns[iradius,imode]*np.sin(angle) + zmnc[iradius,imode]*np.cos(angle)
    
    for imode in range(len(xn_nyq)):
        angle = xm_nyq[imode]*theta2D - xn_nyq[imode]*phi2D
        B = B + bmnc[iradius,imode]*np.cos(angle) + bmns[iradius,imode]*np.sin(angle)
    
    X = R * np.cos(phi2D)
    Y = R * np.sin(phi2D)
    # Rescale to lie in [0,1]:
    B_rescaled = (B - B.min()) / (B.max() - B.min())
    
    # now make a vts file
    filename = f"./data/nfp4_QH_aspect_{aspect_target}"
    
    from pyevtk.hl import gridToVTK
    x=X.reshape((1, ntheta, nphi))
    y=Y.reshape((1, ntheta, nphi))
    z=Z.reshape((1, ntheta, nphi))
    modB=B_rescaled.reshape((1, ntheta, nphi))
    pointData = {'modB':modB}
    gridToVTK(filename, x, y, z, pointData=pointData)
    
    
    
