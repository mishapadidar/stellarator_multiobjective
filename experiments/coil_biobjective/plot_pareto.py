import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
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

# paraview files
paraview_files = ["./output/biobjective/length/biobjective_eps_con_length_14.0_cold_ncoils_4_45d6612c-1c84-4c56-b5cc-69aba4906332.pickle",
    "./output/biobjective/length/biobjective_eps_con_length_19.333333333333332_warm_ncoils_4_139bd8e6-79ad-4c3b-84b2-216ba1449ee8.pickle"]


# find the files
filelist = glob.glob("./output/biobjective/length/biobjective_eps_con_length_*.pickle")
# filelist = glob.glob("./output_original/biobjective/length/biobjective_eps_con_length_*.pickle")

Fopt_list = []
constraint_targets = []
ncoils = []
for ff in filelist:
    indata = pickle.load(open(ff,"rb"))
    Fopt_list.append(indata['Fopt'])
    constraint_targets.append(indata['constraint_target'])
    ncoils.append(indata['ncoils'])
Fopt_list = np.array(Fopt_list)
constraint_targets = np.array(constraint_targets)
ncoils = np.array(ncoils)
filelist = np.array(filelist)

surf_minor_radius = 0.16831206437162438
surf_effective_circumference = 2*np.pi*surf_minor_radius

# filter out points with small/large average coil length
idx_filter = (Fopt_list[:,1]/ncoils > 3.5) & (Fopt_list[:,1]/ncoils < 8)
Fopt_list = Fopt_list[idx_filter]
constraint_targets = constraint_targets[idx_filter]
ncoils = ncoils[idx_filter]
filelist = filelist[idx_filter]

# filter out points with 3 or 5 coils
idx_filter = ncoils == 4
Fopt_list = Fopt_list[idx_filter]
constraint_targets = constraint_targets[idx_filter]
ncoils = ncoils[idx_filter]
filelist = filelist[idx_filter]

fig,ax = plt.subplots(figsize=(8,8))
ncoils_unique = np.unique(ncoils)

# get the Pareto set
for ii,nn in enumerate(ncoils_unique):
    # find the pareto set for this number of coils
    idx_nn = ncoils == nn
    idx_pareto = is_pareto_efficient(Fopt_list[idx_nn])
    F_pareto = Fopt_list[idx_nn][idx_pareto]
    
    # plot samples
#     plt.scatter(Fopt_list[idx_nn,1]/nn/surf_effective_circumference,Fopt_list[idx_nn,0],color=colors[ii],marker='o',s=15,label='Samples, $n_c = %d$'%nn)

    # plot pareto for total coil length
#     plt.scatter(F_pareto[:,1],F_pareto[:,0],color=colors[ii],marker=markers[ii],s=80,label='$n_c = %d$'%nn)

    # plot pareto for average coil length
#     plt.scatter(F_pareto[:,1],F_pareto[:,0],color=colors[ii],marker=markers[ii],s=80,label='$n_c = %d$'%nn)
    plt.scatter(F_pareto[:,1]/nn/surf_effective_circumference,F_pareto[:,0],color='k',marker=markers[ii],s=50,zorder=99)

    # Make quadratic flux have units [T]
#     plt.scatter(F_pareto[:,1]/nn/surf_effective_circumference,np.sqrt(F_pareto[:,0])/surf_minor_radius,color=colors[ii],marker=markers[ii],s=80,label='$%d$ coils'%nn)

# plot the paraview coils
idx_paraview_files= np.array([ii for ii,xx in enumerate(filelist) if xx in paraview_files])
markers= ['*','s','d']
for ii,jj in enumerate(idx_paraview_files):
    plt.scatter(Fopt_list[jj,1]/nn/surf_effective_circumference,Fopt_list[jj,0],color=colors[ii],marker=markers[ii],s=200,zorder=100)


plt.ylabel('Quadratic Flux')
plt.xlabel("Average Coil Length / Surface Minor Circumference")
#plt.legend(loc='upper right')

# add y-ticks to show the grid
plt.yscale('log')
plt.yticks([3.162e-6,1e-6,3.162e-7,1e-7])
# plt.xlim(20,30)

# grid
plt.grid(linewidth=2,zorder=1)


# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()


filename = "./output/biobjective_pareto_plot.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")

# copy the files of pareto points to another directory
cp_pareto_files = True
if cp_pareto_files:
    for ii,nn in enumerate(ncoils_unique):
        
        # find the pareto set for this number of coils
        idx_nn = ncoils == nn
        idx_pareto = is_pareto_efficient(Fopt_list[idx_nn])
        pareto_files = filelist[idx_nn][idx_pareto]
        
        # copy the pareto files to another dir
        paretodir = "./output/pareto_data"
        if not os.path.exists(paretodir):
            os.makedirs(paretodir)
            
        for src in pareto_files:
            # copy the pickle
            tag = src.split("/")[-1]
            shutil.copyfile(src, paretodir+"/"+tag)
            # copy the vtu
            src = src[:-6] + "vtu"
            tag = tag[:-6] + "vtu"
            shutil.copyfile(src, paretodir+"/"+tag)
    
