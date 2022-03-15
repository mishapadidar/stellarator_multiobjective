import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')
import pickle

"""
plot QS^2 error via aspect
"""

# TODO: get new files for aspect=3,6,8
filelist = ['./data/data_aspect_3.0_20220313214007.pickle','./data/data_aspect_4.0_20220313210157.pickle','./data/data_aspect_5.0_20220313214020.pickle','./data/data_aspect_6.0_20220314093352.pickle','./data/data_aspect_7.0_20220313213937.pickle']

aspect_list = []
qs_list  = []
for ff in filelist:
  indata = pickle.load(open(ff,"rb"))
  residuals = indata['residuals']
  qs_mse = np.mean(residuals[:-1]**2)
  target = indata['aspect_target']
  asp = residuals[-1] + target
  print('qs mse: ',qs_mse,'asp',asp)
  aspect_list.append(asp)
  qs_list.append(qs_mse)
plt.plot(aspect_list,qs_list,'-o',linewidth=3)
plt.xlabel('aspect ratio')
plt.ylabel('Quasisymmetry MSE')
plt.yscale('log')
plt.show()
  



