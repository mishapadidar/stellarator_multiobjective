import numpy as np
import sys
import pickle
import subprocess

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
  
  
if __name__=="__main__":  
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
