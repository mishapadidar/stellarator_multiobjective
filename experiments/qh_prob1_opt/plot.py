import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')
import pickle
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

"""
plot QS^2 error via aspect
"""

plot_all_points = True

# load data
indata = pickle.load(open("./data/function_values.pickle","rb"))
aspect_list = indata['aspect']
qs_list = indata['qs_mse']
# truncate plot to [3,10]
idx_trunc = np.logical_and(aspect_list>=3,aspect_list<=10)
aspect_list = aspect_list[idx_trunc]
qs_list = qs_list[idx_trunc]
# stack it
FX = np.vstack((aspect_list,qs_list)).T
# make pareto set
idx_pareto = is_pareto_efficient(FX)
aspect_pareto = FX[idx_pareto,0]
qs_pareto = FX[idx_pareto,1]


# plot
if plot_all_points:
  plt.scatter(aspect_list,qs_list,alpha=0.5)
plt.scatter(aspect_pareto,qs_pareto,s=20,color='r',label='pareto')
plt.xlabel('aspect ratio')
plt.ylabel('Quasisymmetry MSE')
plt.yscale('log')
plt.legend()
plt.show()
  



