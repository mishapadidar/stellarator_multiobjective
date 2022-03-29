
import numpy as np
import os
import pickle
import glob 
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

def find_pareto_front(datadir,save=False):
  """
  Find the pareto front over a given set of files. 

  datadir: directory holding the pickle files
           should be formatted as i.e. "./data"
  """

  filelist = glob.glob(datadir +"/data*.pickle")

  # read data
  X = []
  aspect_list = np.zeros(0)
  qs_list  = np.zeros(0)
  for ff in filelist:
    # load data
    print('loading',ff)
    indata = pickle.load(open(ff,"rb"))
    RX = indata['RawX']
    qs_mse = np.mean(RX[:,:-1]**2,axis=1)
    asp = RX[:,-1]

    # append new data to lists
    for xx in indata['X']:
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
    pickle.dump(outdata,open(datadir + "/pareto_optimal_points.pickle","wb"))
  return X,FX


if __name__=="__main__":
  print(find_pareto_front("./data",save=True))
