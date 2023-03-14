
import numpy as np
import os
import pickle
import glob 
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

def find_pareto_front(datadir,save=True,from_scratch=True,asp_lb=3.0,asp_ub=10.0):
  """
  Find the pareto front over a given set of files. 

  datadir: directory holding the pickle files
           should be formatted as i.e. "./data"
  save: save the pareto front data
  from_scratch: compute the pareto front from scratch. Use this
   if you have overwritten any old files.
  """

  filelist = glob.glob(datadir +"/data*.pickle")
  filelist.sort()

  outfilename = datadir + "/pareto_optimal_points.pickle"
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
    RX = indata['RawX']
    qs_mse = np.mean(RX[:,:-1]**2,axis=1)
    asp = RX[:,-1]

    # truncate data to [3,10]
    idx_trunc = np.logical_and(asp>=asp_lb,asp<=asp_ub)
    asp = asp[idx_trunc]
    qs_mse = qs_mse[idx_trunc]
    inX = indata['X'][idx_trunc]

    # append new data to lists
    #for xx in indata['X']:
    for xx in inX:
      X.append(xx)
    aspect_list = np.append(aspect_list,asp)
    qs_list = np.append(qs_list,qs_mse)

    # compute pareto set
    FX = np.vstack((aspect_list,qs_list)).T
    idx_pareto = is_pareto_efficient(FX,return_mask=False)

    # only keep pareto points
    aspect_list = aspect_list[idx_pareto]
    qs_list = qs_list[idx_pareto]
    FX = FX[idx_pareto]
    X  = [X[ii] for ii in idx_pareto]

    if save:
      outdata = {}
      outdata['X'] = X
      outdata['FX'] = FX
      outdata['filelist'] = processed
      pickle.dump(outdata,open(datadir + "/pareto_optimal_points.pickle","wb"))
  return X,FX


if __name__=="__main__":
  find_pareto_front("./data",save=True,from_scratch=True)
