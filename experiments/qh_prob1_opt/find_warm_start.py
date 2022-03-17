import numpy as np
import glob 

def find_warm_start(aspect_target,dir_list,thresh =1e-4):
  assert (thresh <= 10.0 and thresh >= 0.0)

  filelist = []
  # find files
  for dd in dir_list:
    filelist += glob.glob(dd+"/*.pickle")

  # read data
  X = []
  QX = []
  for ff in filelist:
    indata = pickle.load(open(ff,"rb"))
    xopt = indata['xopt']
    asp = indata['residuals'][-1] + indata['aspect_target']
    qs_mse = np.mean(indata['residuals'][:-1]**2)
    if abs(asp-asp_target)<=thresh:
      X.append(xopt)
      Q.append(qs_mse)

  if len(QX) == 0:
    # restart
    return find_warm_start(aspect_target,dir_list,10*thresh):
  else:
    # choose the best point within the threshold
    return np.argmin(QX)
  
