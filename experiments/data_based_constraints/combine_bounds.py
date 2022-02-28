
import pickle
import numpy as np
"""
A function to combine the bounds in many pickle files.

Run this script with `python3 ...` to compute the bounds
from all of the data files and dump a pickle file with
the combined bounds
"""

def combine_bounds_from_files(filelist):
  """
  computes largest bounds from a set of bounds
  """
  for ii,ff in enumerate(filelist):
    indata = pickle.load(open(ff,"rb"))
    if ii == 0:
      lb = indata['lb']
      ub = indata['ub']
    else:
      lb_f = indata['lb']
      ub_f = indata['ub']
      lb = np.copy(np.minimum(lb,lb_f))
      ub = np.copy(np.maximum(ub,ub_f))
  return lb,ub

if __name__=="__main__":
  import glob
  filelist = glob.glob("./data/samples_*.pickle")
  lb,ub = combine_bounds_from_files(filelist)
  d = {'lb':lb,'ub':ub}
  pickle.dump(d, open("../../problem/bounds.nfp4_QH_warm_start_high_res.pickle","wb"))
