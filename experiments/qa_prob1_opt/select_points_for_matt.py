
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

max_modes = {8:1,24:2,48:3,80:4,120:5}
vmec_input = "../../problem/input.nfp2_QA_high_res"


for idx in range(len(X)):
  x0 = X[idx]
  max_mode = max_modes[len(x0)]
  if max_mode < 5:
    continue
  prob = qa_prob1.QAProb1(max_mode=max_mode,vmec_input = vmec_input)
  # evaluate the point
  raw = prob.raw(x0)

  print('max_mode',max_mode)
  print('qs_mse',qs[idx],'aspect',aspect[idx])
  qs_mse = np.mean(raw[:-2]**2)
  print('qs_mse',qs_mse,'aspect',raw[-2])
  print("")

  # get the wout file
  wout_files = glob.glob("wout_nfp2_QA_high_res_*.nc")
  assert len(wout_files) == 1
  os.rename(wout_files[0], f"points_for_matt/wout_QA_nfp2_qs_mse_{qs_mse}_aspect_{raw[-2]}_max_mode_{max_mode}.nc")
