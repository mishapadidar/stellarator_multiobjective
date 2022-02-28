import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')

filelist = glob.glob("./data/samples_*.pickle")
dim_x = 24
dim_F = 2
X = np.zeros((0,dim_x)) # points
FX = np.zeros((0,dim_F)) # function values
CX = np.zeros((0,1)) # constraint values

for ii,ff in enumerate(filelist):
  indata = pickle.load(open(ff,"rb"))
  FY = indata['FX']
  CY = indata['CX']
  FX = np.copy(np.vstack((FX,FY)))
  CX = np.copy(np.vstack((CX,CY)))

#colors = cm.jet(CX.flatten())
plt.scatter(FX[:,0],FX[:,1],c = CX.flatten())
plt.ylabel('Aspect')
plt.xlabel('QS')
#plt.yscale('log')
plt.legend()
plt.show()
