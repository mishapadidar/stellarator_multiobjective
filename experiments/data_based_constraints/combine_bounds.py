
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
      #F_lb = np.min(indata['FX'],axis=0)
      F_ub = np.max(indata['FX'],axis=0)
    else:
      lb_f = indata['lb']
      ub_f = indata['ub']
      lb = np.copy(np.minimum(lb,lb_f))
      ub = np.copy(np.maximum(ub,ub_f))

      #lb_temp = np.min(indata['FX'],axis=0)
      ub_temp = np.max(indata['FX'],axis=0)
      #F_lb = np.copy(np.minimum(lb_temp,F_lb)) # analytic lower bound
      F_lb = np.zeros_like(F_ub)
      F_ub = np.copy(np.maximum(ub_temp,F_ub))
     
  return lb,ub,F_lb,F_ub

if __name__=="__main__":
  import glob
  filelist = glob.glob("./data/samples_*.pickle")
  lb,ub,F_lb,F_ub = combine_bounds_from_files(filelist)
  d = {'lb':lb,'ub':ub,'F_lb':F_lb,'F_ub':F_ub}
  print(d)
  pickle.dump(d, open("../../problem/bounds.nfp4_QH_warm_start_high_res.pickle","wb"))
