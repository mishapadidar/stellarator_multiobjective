import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
import glob

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})
#import matplotlib.style as style
#style.use('tableau-colorblind10')

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']
markers = ['s','o','^','x','d']
linestyles=['solid'
            ,'solid'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashdot'
            ,'dashdot'
            ,'dashdot'
            ]

# load the data
infile = "./loss_profile_data.pickle"
indata = pickle.load(open(infile,"rb"))
# load the relevant keys
for key in list(indata.keys()):
    s  = f"{key} = indata['{key}']"
    exec(s)
n_configs = len(filelist)

# skip any configs?
skip = []

# compute the loss profiles
times = np.logspace(-5,np.log10(tmax),1000)
lp_surf = np.array([np.mean(c_times_surface< t,axis=1) for t in times])

# make a figure
fig, ax_both = plt.subplots(figsize=(14,6),ncols=2)
ax1,ax2 = ax_both

# choose colors
#from matplotlib.pyplot import cm
#colors = cm.jet(np.linspace(0, 1, n_configs))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette("colorblind",256))
colors = cmap(np.linspace(0,1,n_configs))

# plot the data
for ii in range(n_configs):
    if ii in skip:
        continue
    label = aspect_list[ii]
    ax1.plot(times,lp_surf[:,ii],linewidth=3,linestyle=linestyles[ii],color=colors[ii])
    ax2.scatter(aspect_list[ii],lp_surf[-1,ii], color='k',s=50)
    print(label,'surface losses',lp_surf[-1,ii])

# legend
#ax1.legend(ncols=3,fontsize=16,frameon=False)
#ax2.legend(ncols=3,fontsize=16,frameon=False)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,bbox_to_anchor=(0.092, 0.93,0.90,0.93),loc=3,
          fancybox=True, shadow=False, ncol=7,fontsize=17,labelspacing=0.5,
           columnspacing=1.15)

for ax in ax_both:
    # darken the border
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('2')  

# log space 
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_yscale('log')
# limits
ax1.set_ylim([1e-4,0.3])
ax1.set_xlim([1e-5,1e-2])
# ticks
ax1.set_xticks([1e-5,1e-4,1e-3,1e-2,1e-1])
ax1.set_yticks([1e-3,1e-2,1e-1])
# labels
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel("Fraction of alpha particles lost",fontsize=18)
ax2.set_xlabel('Aspect Ratio')
ax2.set_ylabel("Fraction of alpha particles lost",fontsize=18)

plt.tight_layout()


filename = "loss_profile.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
