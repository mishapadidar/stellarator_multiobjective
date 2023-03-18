import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
import glob
import os
import shutil
import sys
sys.path.append("../../utils")
from is_pareto_efficient import is_pareto_efficient

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']
markers = ['o','s','^']

infile = "./output/qfm_data.pickle"
indata = pickle.load(open(infile,"rb"))
# load the relevant keys
for key in list(indata.keys()):
    s  = f"{key} = indata['{key}']"
    exec(s)

surf_minor_radius = 0.16831206437162438
surf_effective_circumference = 2*np.pi*surf_minor_radius
ncoils = 4

# coil lengths
lengths = Fopt_list[:,1]

fig,ax = plt.subplots(figsize=(8,8))

ax.scatter(lengths/ncoils/surf_effective_circumference,qsrr_list,color='k',s=50,zorder=100)

# labels
ax.set_xlabel("Average Coil Length / Surface Minor Circumference")
ax.set_ylabel("Departure from Quasi-symmetry $Q_{1,0}$")

# axis scale
ax.set_yscale('log')

# grid
plt.grid(linewidth=2,zorder=1)

# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()

filename = "./output/qfm_plot.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
