import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import ticker
import matplotlib
import pandas as pd
import seaborn as sns
import glob

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
markers = ['s','o','^','x']
linestyles=['-','--','-.']

# matplotlib.use('TkAgg')

filelist = ["./output/biobjective_eps_con_length_14.0_cold_ncoils_4_45d6612c-1c84-4c56-b5cc-69aba4906332_field_strength.pickle","./output/biobjective_eps_con_length_19.333333333333332_warm_ncoils_4_139bd8e6-79ad-4c3b-84b2-216ba1449ee8_field_strength.pickle"]


fig,ax = plt.subplots(figsize=(8,4),ncols=2)

for ifile, infile in enumerate(filelist):
    # load the data
    field_line_data = pickle.load(open(infile,"rb"))
    for key in list(field_line_data.keys()):
        s  = f"{key} = field_line_data['{key}']"
        exec(s)
    
    # plot the data
    p = ax[ifile].contour(zeta_mesh,theta_mesh,modB_list[3].reshape((nzeta,ntheta)),levels=13,linewidths=2)

    # set the colobar
    tick_font_size = 16
    cbar = plt.colorbar(p,ax=ax[ifile],format="%.2f")
    cbar.locator = ticker.MaxNLocator(nbins=6)
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.update_ticks()
    cbar.lines[0].set_linewidth(18)

# Set the ticks and ticklabels for all axes
plt.setp(ax, xticks=[0,np.pi], xticklabels=['0','$\pi$'],
        yticks=[0,2*np.pi], yticklabels=['0','2$\pi$'])


for a in ax:
    # make the axis labels
    a.set_xlabel('$\zeta$')
    a.set_ylabel('$\\theta$')
    a.xaxis.set_label_coords(.5,-0.025)
    a.yaxis.set_label_coords(-0.025,.5)

    # darken the border
    a.patch.set_edgecolor('black')  
    a.patch.set_linewidth(2)  

plt.tight_layout()
#filename = infile.split("/")[-1] # remove any directories
#filename = filename[:-7] # remove the .pickle
#filename = filename + ".pdf"
filename = "./output/biobjective_coils_field_strength_contours.pdf"
print(filename)
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
