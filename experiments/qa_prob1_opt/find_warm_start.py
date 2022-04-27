
import numpy as np
import os
import pickle
import glob 
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

def find_warm_start(datadir,save=True,from_scratch=True):
  """
  Find a set of warm start point over a given set of files.
  Collects the optima from each run into a list.

  datadir: directory holding the pickle files
           should be formatted as i.e. "./data"
  save: save the pareto front data
  from_scratch: compute the point set from scratch. Use this
   if you have overwritten any old files.
  """

  filelist = glob.glob(datadir +"/data*.pickle")
  filelist.sort()

  outfilename = datadir + "/warm_start_points.pickle"
  if os.path.exists(outfilename) and from_scratch is False:
    indata = pickle.load(open(outfilename,"rb"))
    X = indata['X']
    FX = indata['FX']
    aspect_list = np.copy(FX[:,0])
    qs_list = np.copy(FX[:,1])
    processed = indata['filelist']
  else:
    X = []
    aspect_list = np.zeros(0)
    qs_list  = np.zeros(0)
    processed = []

  for ff in filelist:

    if ff in processed:
      continue
    else:
      processed.append(ff)

    # load data
    print('loading',ff)
    indata = pickle.load(open(ff,"rb"))
    xopt = indata['xopt']
    qs_mse_opt = outdata['qs_mse_opt'] 
    aspect_opt = outdata['aspect_opt']

    # append new data to lists
    X.append(xopt)
    aspect_list = np.append(aspect_list,aspect_opt)
    qs_list = np.append(qs_list,qs_mse_opt)

    # make FX
    FX = np.vstack((aspect_list,qs_list))
    if save:
      outdata = {}
      outdata['X'] = X
      outdata['FX'] = FX
      outdata['filelist'] = processed
      pickle.dump(outdata,open(datadir + "/pareto_optimal_points.pickle","wb"))
  return X,FX


if __name__=="__main__":
  find_warm_start("./data",save=True,from_scratch=True)
