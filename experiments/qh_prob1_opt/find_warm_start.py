import numpy as np
import os
import pickle
import glob 

def find_warm_start(aspect_target,dir_list,thresh =5e-4):
  assert (thresh <= 10.0 and thresh >= 0.0)

  filelist = []
  # find files
  for dd in dir_list:
    filelist += glob.glob(dd+"/reduced*.pickle")

  # read data
  X = []
  QX = []
  for ff in filelist:
    # skip some files
    indata = pickle.load(open(ff,"rb"))
    xopt = indata['xopt']
    asp = indata['aspect_opt']
    qs_mse = indata['qs_mse_opt']
    if abs(asp-aspect_target)<=thresh:
      X.append(xopt)
      QX.append(qs_mse)

  if len(QX) == 0:
    # restart
    return find_warm_start(aspect_target,dir_list,10*thresh)
  else:
    # choose the best point within the threshold
    return X[np.argmin(QX)]
  
if __name__=="__main__":
  print(find_warm_start(5.2,["./data"],thresh=5e-4))
