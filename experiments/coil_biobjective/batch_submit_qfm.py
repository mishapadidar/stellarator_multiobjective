import glob 
import subprocess
import shutil
import os
import stat

filelist = glob.glob("./output/pareto_data/*_ncoils_4_*.pickle")

for ii, fname in enumerate(filelist):
  
    # make a submit directory
    subdir = f"./_batch_qfm_{ii}"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # copy the run file
    shutil.copyfile("make_qfm_surface_data.py", subdir + "/make_qfm_surface_data.py")

    # write the submit file
    subname=subdir + f"/submit_qfm.sub"
    subfile = open(subname,"w")

    subfile.write("#!/bin/bash")
    subfile.write("\n")
    subfile.write(f"#SBATCH -J qfm_{ii} # Job name")
    subfile.write("\n")
    subfile.write(f"#SBATCH -o job_%j.out    # Name of stdout output file(%j expands to jobId)")
    subfile.write("\n")
    subfile.write(f"#SBATCH -e job_%j.err    # Name of stderr output file(%j expands to jobId)")
    subfile.write("\n")
    subfile.write(f"#SBATCH -N 1       # Total number of nodes requested")
    subfile.write("\n")
    subfile.write(f"#SBATCH -n 1       # Total number of cores requested")
    subfile.write("\n")
    subfile.write(f"#SBATCH --ntasks-per-node 1    # Total number of cores requested")
    subfile.write("\n")
    subfile.write(f"#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment")
    subfile.write("\n")
    subfile.write(f"#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)")
    subfile.write("\n")
    subfile.write(f"#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU")
    subfile.write("\n")
    subfile.write(f"#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99]")
    subfile.write("\n")
    subfile.write("\n")
    subfile.write(f"mpiexec -n 1 python3 make_qfm_surface_data.py {fname}")
    subfile.write("\n")

    subfile.close()

    ## write a run file
    #runname = subdir + "/run.sh"
    #runfile = open(runname,"w")
    #runfile.write("sbatch --requeue submit_qfm.sub\n")
    #runfile.close()
    #os.chmod(runname, stat.S_IRWXU)
    
    # submit
    os.chdir(subdir)
    subprocess.Popen(f"sbatch --requeue submit_qfm.sub",shell=True)
    os.chdir("../")
    #subprocess.Popen(f"./{runname}",shell=True)

