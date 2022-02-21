import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


infile = "1dplot_data.pickle"
indata = pickle.load(open(infile,'rb'))
T = indata['T']
FX = indata['FX']
n_directions = indata['n_directions']

# which objective to plot
obj_number = 0

for ii in range(n_directions):
  plt.plot(T,FX[ii][:,obj_number],linewidth=2,label=f'dir {ii}')
plt.ylabel('function value')
plt.xlabel('distance from x0')
plt.show()
