
import numpy as np
import pickle
import sys
sys.path.append("../../problem")
sys.path.append("../../utils")
import qa_prob1
import glob
import os
"""
Select pareto points for matt
"""

# load data
infilename = "./data/pareto_optimal_points.pickle"
indata = pickle.load(open(infilename,"rb"))
X = indata['X']
FX = indata['FX']
aspect = np.copy(FX[:,0])
qs = np.copy(FX[:,1])

max_mode = 4
vmec_input = "../../problem/input.nfp2_QA_high_res"
prob = qa_prob1.QAProb1(max_mode=max_mode,vmec_input = vmec_input)


for idx in range(len(X)):
  """
  Resetting the problem helps avoid failures!!!

  If a point cannot be evaluated try using
  mpol=ntor=6,7,8,9,10.
  """
  prob = qa_prob1.QAProb1(max_mode=max_mode,vmec_input = vmec_input)
  x0 = X[idx]
  print('qs_mse',qs[idx],'aspect',aspect[idx])
  # evaluate the point
  # try to get rid of failures
  prob.vmec.indata.mpol = 7
  prob.vmec.indata.ntor = 7
  raw = prob.raw(x0)
  print('qs_mse',np.mean(raw[:-1]**2),'aspect',raw[-1])

  # if fail: remove the wout, and evaluate again
  if np.mean(raw[:-1]**2) == np.inf:
    wout_files = glob.glob("wout_nfp2_QA_high_res_*.nc")
    assert len(wout_files) == 1
    os.remove(wout_files[0])
    prob.vmec.indata.mpol = 8
    prob.vmec.indata.ntor = 8
    raw = prob.raw(x0)
    print('qs_mse',np.mean(raw[:-1]**2),'aspect',raw[-1])

  # get the wout file
  wout_files = glob.glob("wout_nfp2_QA_high_res_*.nc")
  assert len(wout_files) == 1

  # if success: move the wout file
  if np.mean(raw[:-1]**2) < np.inf:
    os.rename(wout_files[0], f"points_for_matt/wout_QA_nfp2_aspect_{raw[-1]}.nc")
  else:
    # remove the wout file
    os.remove(wout_files[0])
