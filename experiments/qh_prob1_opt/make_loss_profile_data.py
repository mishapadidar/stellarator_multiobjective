import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import glob
import pickle
import sys
sys.path.append("../../utils")
from trace_boozer import TraceBoozer
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Make data for the loss profile plots.

usage:
  mpiexec -n 1 python3 make_loss_profile_data.py

We rescale all configurations to the same minor radius and same volavgB.
Scale the device so that the major radius is 
  R = aspect*target_minor
where aspect is the current aspect ratio and target_minor is the desired
minor radius.
"""
# scaling params
target_minor_radius = 1.7
target_B00_on_axis = 5.7

# tracing parameters
n_particles = 5000
tmax = 0.1
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 32
bri_ntor = 32
n_partitions = 1


filelist = glob.glob("./data/input.*")
n_configs = len(filelist)

# storage arrays
c_times_surface = -np.inf*np.ones((n_configs,n_particles))
aspect_list = np.zeros(n_configs)

# for saving data
outfile = "./loss_profile_data.pickle"
outdata = {}
outdata['filelist'] = filelist
outdata['target_minor_radius'] =target_minor_radius
#outdata['target_volavgB'] = target_volavgB
outdata['target_B00_on_axis'] = target_B00_on_axis
outdata['n_particles'] = n_particles
outdata['tmax'] = tmax
outdata['tracing_tol'] = tracing_tol
outdata['interpolant_degree'] = interpolant_degree
outdata['interpolant_level'] =  interpolant_level
outdata['bri_mpol'] = bri_mpol
outdata['bri_ntor'] = bri_ntor

for ii,infile in enumerate(filelist):

  vmec_input = infile

  mpi = MpiPartition(n_partitions)
  vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
  surf = vmec.boundary
 
  # get the aspect ratio for rescaling the device
  aspect_ratio = surf.aspect_ratio()
  major_radius = target_minor_radius*aspect_ratio

  # build a tracer object
  tracer = TraceBoozer(vmec_input,
                        n_partitions=n_partitions,
                        max_mode=-1,
                        aspect_target=aspect_ratio,
                        major_radius=major_radius,
                        target_volavgB=5.0, # dummy value
                        tracing_tol=tracing_tol,
                        interpolant_degree=interpolant_degree,
                        interpolant_level=interpolant_level,
                        bri_mpol=bri_mpol,
                        bri_ntor=bri_ntor)
  tracer.sync_seeds()
  x0 = tracer.x0

  # compute the boozer field
  field,bri = tracer.compute_boozer_field(x0)

  if field is None:
    # boozXform failed
    if rank == 0:
      print("vmec failed for ",infile)
    continue
  
  # now scale the toroidal flux by B(0,0)[s=0]
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array
  # scale the toroidal flux
  tracer.vmec.indata.phiedge *= target_B00_on_axis/B00

  # re-compute the boozer field
  tracer.vmec.need_to_run_code = True
  tracer.vmec.run()
  tracer.field = None # so the boozXform recomputes
  field,bri = tracer.compute_boozer_field(x0)

  # now get B00 just to make sure it was set right
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array

  # also get the minor radius
  major_rad = tracer.surf.get("rc(0,0)")
  aspect = tracer.surf.aspect_ratio()
  minor_rad = major_rad/aspect

  
  if rank == 0:
    print("")
    print("processing", infile)
    print('aspect',aspect)
    print('minor radius',minor_rad)
    print("axis B00",B00)
    print('volavgB',tracer.vmec.wout.volavgB)
    print('toroidal flux',tracer.vmec.indata.phiedge)
  
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,0.25)
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  if rank == 0:
    lf = np.mean(c_times < tmax)
    print('surface loss fraction:',lf)
    print("")

  # store the data
  c_times_surface[ii] = np.copy(c_times)
  aspect_list[ii] = aspect

  # save the data
  outdata['c_times_surface'] = c_times_surface
  outdata['aspect_list'] = aspect_list
  pickle.dump(outdata,open(outfile,"wb"))
