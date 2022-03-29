import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
#matplotlib.use('TkAgg')
import pickle
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

"""
plot QS^2 error via aspect
"""

filelist = glob.glob("./data/*.pickle")

aspect_list = np.zeros(0)
qs_list  = np.zeros(0)
for ff in filelist:
  print('loading',ff)
  indata = pickle.load(open(ff,"rb"))
  RX = indata['RawX']
  qs_mse = np.mean(RX[:,:-1]**2,axis=1)
  asp = RX[:,-1]
  aspect_list = np.append(aspect_list,asp)
  qs_list = np.append(qs_list,qs_mse)
  FX = np.vstack((aspect_list,qs_list)).T
  idx_pareto = is_pareto_efficient(FX)
  aspect_list = aspect_list[idx_pareto]
  qs_list = qs_list[idx_pareto]

print('computing pareto set')
# comppute pareto set
FX = np.vstack((aspect_list,qs_list)).T
idx_pareto = is_pareto_efficient(FX)
aspect_pareto = FX[idx_pareto,0]
qs_pareto = FX[idx_pareto,1]
idx_sort = np.argsort(aspect_pareto)
aspect_pareto = aspect_pareto[idx_sort]
qs_pareto = qs_pareto[idx_sort]

print('plotting')
plt.scatter(aspect_list,qs_list,alpha=0.5)
plt.plot(aspect_pareto,qs_pareto,'-s',color='r',linewidth=3)
plt.xlabel('aspect ratio')
plt.ylabel('Quasisymmetry MSE')
plt.yscale('log')
plt.show()
  



