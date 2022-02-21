import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


infile = "2dplot_data_6_15.pickle"
indata = pickle.load(open(infile,'rb'))
X1 = indata['X1']
X2 = indata['X2']
idx1 = indata['idx1']
idx2 = indata['idx2']
FX = indata['FX']

# which objective to plot
obj_number = 0

plt.contour(X1,X2,FX[obj_number])
plt.xlabel(f'x_{idx1}')
plt.ylabel(f'x_{idx2}')
plt.title(f"Objective {obj_number} over 2d-slice")
plt.legend()
plt.show()
