import pickle
import numpy as np
import glob

# load the data
s_list = [0.25,0.5]
for s_label in s_list:
    infile = f"./loss_profile_data_s_{s_label}.pickle"
    indata = pickle.load(open(infile,"rb"))
    # load the relevant keys
    for key in list(indata.keys()):
        s  = f"{key} = indata['{key}']"
        exec(s)
    n_configs = len(filelist)
    
    loss_frac = np.mean(c_times_surface < tmax,axis=1)
    
    idx_sort = np.argsort(aspect_list)
    aspect_list = aspect_list[idx_sort]
    loss_frac = loss_frac[idx_sort]
    aspect_str = "Aspect & "
    loss_str = f"s = {s_label} & "
    for ii,x in enumerate(aspect_list):
        aspect_str += "%.2f"%x
        loss_str += "%.4f"%loss_frac[ii]
        if ii < len(aspect_list)-1:
            aspect_str += " & "
            loss_str += " & "
    print("")
    print('surface', s_label)
    print(aspect_str)
    print(loss_str)
