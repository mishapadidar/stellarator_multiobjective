import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')


infile = "1dplot_data.pickle"
indata = pickle.load(open(infile,'rb'))
T = indata['T']
FX = indata['FX']
n_directions = indata['n_directions']

# which objective to plot
obj_number = 0

#colors = cm.rainbow(np.linspace(0, 1, n_directions))
colors = cm.jet(np.linspace(0, 1, n_directions))
for ii in range(n_directions):
  plt.plot(T,FX[ii][:,obj_number],linewidth=2,color=colors[ii],label=f'dir {ii}')
plt.ylabel('function value')
plt.xlabel('distance from x0')
#plt.xscale('symlog',linthresh=1e-9)
plt.yscale('log')
plt.legend()
plt.show()
