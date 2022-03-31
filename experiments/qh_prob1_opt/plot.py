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

plot_all_optima = False
if plot_all_optima:
  filelist = glob.glob("./data/data*.pickle")
  aspect_list = np.zeros(0)
  qs_list  = np.zeros(0)
  for ff in filelist:
    print('loading',ff)
    indata = pickle.load(open(ff,"rb"))
    aspect_list = np.append(aspect_list,indata['aspect_opt'])
    qs_list = np.append(qs_list,indata['qs_mse_opt'])
  FX = np.vstack((aspect_list,qs_list)).T

# load pareto set
infilename = "./data/pareto_optimal_points.pickle"
indata = pickle.load(open(infilename,"rb"))
FX_pareto = indata['FX']
aspect_pareto = np.copy(FX_pareto[:,0])
qs_pareto = np.copy(FX_pareto[:,1])

# plot
if plot_all_optima:
  plt.scatter(aspect_list,qs_list,alpha=0.5)
plt.scatter(aspect_pareto,qs_pareto,s=20,color='r',label='pareto')
plt.xlabel('aspect ratio')
plt.ylabel('Quasisymmetry MSE')
plt.yscale('log')
plt.legend()
plt.show()
  



