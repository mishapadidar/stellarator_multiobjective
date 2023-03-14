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

import matplotlib
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})

colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']

"""
plot QS^2 error via aspect
"""

plot_all_points = False

# load data
#indata = pickle.load(open("./data/function_values.pickle","rb"))
indata = pickle.load(open("./data/plot_data.pickle","rb"))
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


# paraview points that we are plotting
example_points = np.array([
        [3.499999825854719,4.4651239341053535e-09],
        [5.640672937579123,5.646371127198081e-10],
        [8.75152077871298 ,3.074346514048328e-10]])

# turn qs_mse back to qs ratio residual
qs_pareto = 44352*qs_pareto
example_points[:,1] *= 44352

# plot
fig,ax = plt.subplots(figsize=(8,8))
if plot_all_points:
  plt.scatter(aspect_list,qs_list,alpha=0.5)
#plt.scatter(aspect_pareto,qs_pareto,s=30,color='k',label='pareto front',rasterized=True)
plt.scatter(aspect_pareto,qs_pareto,s=30,color='k',rasterized=True,zorder=100)
plt.scatter(example_points[0,0],example_points[0,1],s=160,color=colors[0],marker='*',zorder=100)
plt.scatter(example_points[1,0],example_points[1,1],s=100,color=colors[1],marker='s',zorder=100)
plt.scatter(example_points[2,0],example_points[2,1],s=120,color=colors[2],marker='d',zorder=100)
plt.xlabel('Aspect Ratio')
plt.ylabel('Quasisymmetry Ratio Residual')
plt.yscale('log')
#plt.legend(loc='upper right')

# set the yticks
plt.yticks([1e-3,3.162e-4,1e-4,3.162e-5,1e-5])
#ticks,labels= plt.yticks()
#labels[2] = ""
#plt.yticks(ticks,labels)

#ax.yaxis.set_label_coords(-0.035,0.5)

plt.grid(linewidth=2,zorder=1)
  
# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()


filename = "aspect_qs_pareto_front.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")

#plt.show()


