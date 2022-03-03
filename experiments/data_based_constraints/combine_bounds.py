
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
    X = indata['X']
    FX = indata['FX']
    CX = indata['CX'].flatten().astype(bool)
    FX = FX[CX]
    X  = X[CX]
    if ii == 0:
      X_ub = np.copy(np.max(X,axis=0))
      X_lb = np.copy(np.min(X,axis=0))
      F_ub = np.copy(np.max(FX,axis=0))
      F_lb = np.copy(np.min(FX,axis=0))
    else:
      X_ub_kp1 = np.copy(np.max(X,axis=0))
      X_lb_kp1 = np.copy(np.min(X,axis=0))
      F_ub_kp1 = np.copy(np.max(FX,axis=0))
      F_lb_kp1 = np.copy(np.min(FX,axis=0))

      X_ub = np.copy(np.maximum(X_ub,X_ub_kp1))
      X_lb = np.copy(np.minimum(X_lb,X_lb_kp1))
      F_ub = np.copy(np.maximum(F_ub_kp1,F_ub))
      F_lb = np.copy(np.minimum(F_lb_kp1,F_lb))
     
  return X_lb,X_ub,F_lb,F_ub

if __name__=="__main__":
  import glob
  filelist = glob.glob("./data/samples_*.pickle")
  lb,ub,F_lb,F_ub = combine_bounds_from_files(filelist)
  d = {'lb':lb,'ub':ub,'F_lb':F_lb,'F_ub':F_ub}
  print(d)
  pickle.dump(d, open("../../problem/bounds.nfp4_QH_warm_start_high_res.pickle","wb"))
