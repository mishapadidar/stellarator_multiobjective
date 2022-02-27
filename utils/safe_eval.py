import numpy as np
import sys
import pickle
import subprocess
from multiprocessing import Pool

class SafeEval():
  """
  A class for performing crash-resilient evaluations of simsopt.
  Evalutations are performed by launching mpi processes through a python
  subprocess call, which read and write simsopt evaluation information to a file,
  to be read by the SafeEval class. 

  This class can only perform seriel evals.
  
  The driver script which calls this class should be run with `python3 ...` not `mpiexec`.
  """

  def __init__(self,dim_F,vmec_input,eval_script = "safe_eval.py",default_F = np.inf):
    """
    dim_F: dimension of function output, expected >1
    vmec_input: vmec input file
    eval_script: string, name of file that performs the evaluation
                  This file, safe_eval.py, contains an evaluation of qh_prob1 and is used
                  as the default eval script. Alternatively you can write your own 
                  eval script by modeling it after the `if __name__=="__main__" portion of 
                  this script.
    default_F: default value to return if an evaluation fails.
    """
    self.dim_F = dim_F
    self.vmec_input = vmec_input
    self.eval_script = eval_script
    self.default_F = default_F*np.ones(dim_F)

  def eval(self,yy):
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
    outdata['F'] = self.default_F
    outdata['vmec_input'] = self.vmec_input
    pickle.dump(outdata,open('_safe_eval.pickle','wb'))
    # subprocess call
    bashCommand = f"mpiexec -n 1 python3 {self.eval_script} _safe_eval.pickle"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # read the pickle
    indata = pickle.load(open('_safe_eval.pickle','rb'))
    return indata['F']
  
  
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
  prob = qh_prob1.QHProb1(vmec_input)
  F = prob.eval(x)
  # write to output
  indata['F'] = F
  pickle.dump(indata,open(infile,"wb"))
