import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import pickle
import sys


"""
Select a couple example pareto optimal points. 
These will be used to make paraview plots and 
B-field contour plots later on.

usage:
  mpiexec -n 1 python3 select_example_points.py
"""
# plot efficient points with these aspect ratios
aspect_list = [3.5,5.5,8.7]

# load the pareto set
infilename = "./data/pareto_optimal_points.pickle"
indata = pickle.load(open(infilename,"rb"))
X = indata['X']
FX = indata['FX']
aspect = np.copy(FX[:,0])
qs = np.copy(FX[:,1])

for aspect_target in aspect_list:

    # find the closest point to the target
    idx = np.argmin(np.abs(aspect - aspect_target))
    x0 = X[idx]
    print('qs_mse',qs[idx],'aspect',aspect[idx])
    
    # evaluate the point
    max_mode = 5
    vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
    n_partitions=1
    mpi = MpiPartition(n_partitions)
    vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
    
    # generate the .wout data
    vmec.run()

    # write an input file
    vmec.write_input(f"./data/input.nfp4_QH_aspect_{aspect_target}")
