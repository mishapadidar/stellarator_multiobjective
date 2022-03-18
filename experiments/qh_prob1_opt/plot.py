import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')
import pickle

"""
plot QS^2 error via aspect
"""

filelist = glob.glob("./data/*.pickle")

aspect_list = np.zeros(0)
qs_list  = np.zeros(0)
for ff in filelist:
  indata = pickle.load(open(ff,"rb"))
  RX = indata['RawX']
  qs_mse = np.mean(RX[:,:-1]**2,axis=1)
  asp = RX[:,-1]
  aspect_list = np.append(aspect_list,asp)
  qs_list = np.append(qs_list,qs_mse)
plt.scatter(aspect_list,qs_list,'-o',linewidth=3)
plt.xlabel('aspect ratio')
plt.ylabel('Quasisymmetry MSE')
plt.yscale('log')
plt.show()
  



