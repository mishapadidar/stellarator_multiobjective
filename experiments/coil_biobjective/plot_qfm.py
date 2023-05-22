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
matplotlib.rcParams.update({'font.size': 20})

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']
markers = ['o','s','^']

filelist = glob.glob("./output/qfm_data/*pickle")
n_configs = len(filelist)

# storage
Fopt_list = np.zeros((n_configs,2))
qsrr_list = np.zeros(n_configs)

# load data
for ifile, infile in enumerate(filelist):
    # load the relevant keys
    indata = pickle.load(open(infile,"rb"))
    for key in list(indata.keys()):
        s  = f"{key} = indata['{key}']"
        exec(s)
    Fopt_list[ifile] = Fopt
    qsrr_list[ifile] = qsrr
    
surf_minor_radius = 0.16831206437162438
surf_effective_circumference = 2*np.pi*surf_minor_radius

# files we plotted in paraview
paraview_files = ["./output/qfm_data/biobjective_eps_con_length_17.625_warm_ncoils_4_55f51166-9fbd-4fc9-b1ec-b463f92e629e_qfm_data.pickle",
"./output/qfm_data/biobjective_eps_con_length_24.189999999999998_warm_ncoils_4_b1775dc0-ea0a-4947-b775-a08d59eacf95_qfm_data.pickle"]
paraview_indexes = [ii for ii,ff in enumerate(filelist) if ff in paraview_files]

# coil lengths
lengths = Fopt_list[:,1]

fig,ax = plt.subplots(figsize=(9,9))

ncoils = 4
ax.scatter(lengths/ncoils/surf_effective_circumference,qsrr_list,color='k',s=50,zorder=99)

# put markers at the paraview points
plt.scatter(lengths[paraview_indexes[0]]/ncoils/surf_effective_circumference,qsrr_list[paraview_indexes[0]],color=colors[0],marker='*',s=250,zorder=100)
plt.scatter(lengths[paraview_indexes[1]]/ncoils/surf_effective_circumference,qsrr_list[paraview_indexes[1]],color=colors[1],marker='s',s=200,zorder=100)

# show the LP-QA quasi-symmetry level
plt.text(3.6,4e-6,"LP-QA quasi-symmetry level",fontsize=20)
ax.axhline(surface_qsrr,color='k',linestyle='--',linewidth=2)

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

filename = "./output/qfm_coils_qs_plot.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
