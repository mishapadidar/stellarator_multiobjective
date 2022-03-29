import pickle
import glob


filelist = []
# find files
indir = "./data/"
filelist += glob.glob(indir + "data_aspect_*.pickle")
for ff in filelist:
  indata = pickle.load(open(ff,"rb"))
  # remove heavy components
  del indata['X']
  del indata['RawX']
  # drop a new file
  outname = ff.split("/")[-1]
  outname = indir + "reduced_" + outname
  print(outname)
  pickle.dump(indata, open(outname,"wb"))
