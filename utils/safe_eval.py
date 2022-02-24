import numpy as np
import sys
import pickle
import subprocess
from multiprocessing import Pool

def safe_eval(yy,dim_F,vmec_input):
  """
  Do a single evaluation safely
  
  1. write a pickle file with yy and vmec_input. Write the default function value as well
  2. call the safe evaluation python script to read that file and do the evaluation
     and write back to the file
  3. read the file and return the evaluation.
  """
  # write a pickle file
  outdata = {}
  outdata['x'] = yy
  outdata['F'] = np.inf*np.ones(dim_F)
  outdata['vmec_input'] = vmec_input
  pickle.dump(outdata,open('safe_eval.pickle','wb'))
  # subprocess call
  bashCommand = "mpiexec -n 1 python3 safe_eval.py safe_eval.pickle"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  # read the pickle
  indata = pickle.load(open('safe_eval.pickle','rb'))
  return indata['F']
  
# define a function for pool
def subpcall(ii):
  # subprocess call
  bashCommand = f"mpiexec -n 1 python3 safe_eval.py _safe_eval_{ii}.pickle"
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()

def safe_evalp(Y,dim_F,vmec_input,n_partitions):
  """
  Do a parallel evaluation safely.
  Y: 2D array of points
  dim_F: dimension of function output, expected >1
  vmec_input: vmec input file
  n_partitions: number of mpi ranks to use for evals.

  Notes:
  The driver script that calls this function CANNOT be run with MPI, 
  i.e. just do python3 driver_script.py. However,
  Make sure that slurm allocates more than n_partitions mpi ranks for the job.
  So include the following in your slurm .submission file 
    #SBATCH -N n_partitions+1
    #SBATCH -n n_partitions+1
  """
  n_points = len(Y)
  # write pickle files
  for ii in range(n_points):
    outdata = {}
    outdata['x'] = Y[ii]
    outdata['F'] = np.inf*np.ones(dim_F)
    outdata['vmec_input'] = vmec_input
    pickle.dump(outdata,open(f'_safe_eval_{ii}.pickle','wb'))
  
  # do the evals with pool
  with Pool(n_partitions) as p:
    p.map(subpcall, range(n_points))

  # storage
  FY = np.zeros((n_points,dim_F))

  # read the pickle files
  for ii in range(n_points):
    indata = pickle.load(open(f'_safe_eval_{ii}.pickle','rb'))
    FY[ii] = np.copy(indata['F'])
  return FY
  
if __name__=="__main__":  
  """
  Here is the actual function evaluation that is performed.
  """ 
  import sys
  import pickle
  sys.path.append("../problem")
  import qh_prob1
  
  # load the point
  infile = sys.argv[1]
  indata = pickle.load(open(infile,'rb'))
  vmec_input = indata['vmec_input']
  x = indata['x']
  # do the evaluation
  prob = qh_prob1.QHProb1(vmec_input,n_partitions=1)
  F = prob.eval(x)
  # write to output
  indata['F'] = F
  pickle.dump(indata,open(infile,"wb"))
