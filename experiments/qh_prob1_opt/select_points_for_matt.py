
import numpy as np
import pickle
import sys
sys.path.append("../../problem")
sys.path.append("../../utils")
import qh_prob1
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

# set a target point aspect
aspect_target =8.7

# find the closest point to the target
idx = np.argmin(np.abs(aspect - aspect_target))
x0 = X[idx]
print('qs_mse',qs[idx],'aspect',aspect[idx])

# evaluate the point
max_mode = 5
vmec_input = "../../problem/input.nfp4_QH_warm_start_high_res"
prob = qh_prob1.QHProb1(max_mode=max_mode,vmec_input = vmec_input)
raw = prob.raw(x0)
print('qs_mse',np.mean(raw[:-1]**2),'aspect',raw[-1])
