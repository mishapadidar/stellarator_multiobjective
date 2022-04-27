
import numpy as np
import os
import pickle
import glob 
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

def collect_function_values(datadir,from_scratch=True):
  """
  Collect the function values into a single pickle file. 
  Script only collects [aspect,qs_mse]. It does not save the residuals
  or any point data.

  datadir: directory holding the pickle files
           should be formatted as i.e. "./data"
  from_scratch: collect the values from scratch. Use this
   if you have overwritten any old files.
  """

  filelist = glob.glob(datadir +"/data*.pickle")
  filelist.sort()

  outfilename = datadir + "/plot_data.pickle"
  if os.path.exists(outfilename) and from_scratch is False:
    indata = pickle.load(open(outfilename,"rb"))
    aspect_list = indata['aspect']
    qs_list = indata['qs_mse']
    processed = indata['filelist']
  else:
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

    # append new data to lists
    aspect_list = np.append(aspect_list,asp)
    qs_list = np.append(qs_list,qs_mse)

    # only keep finite values
    idx_fin = np.isfinite(qs_list)
    aspect_list = aspect_list[idx_fin]
    qs_list = qs_list[idx_fin]
 
    outdata = {}
    outdata['aspect'] = aspect_list
    outdata['qs_mse'] = qs_list
    outdata['filelist'] = processed
    pickle.dump(outdata,open(outfilename,"wb"))
  return aspect_list,qs_list


if __name__=="__main__":
  collect_function_values("./data",from_scratch=False)
