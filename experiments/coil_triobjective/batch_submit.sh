# This script runs a set of biobjective jobs
# each core runs one epsilon constraint solve

constraint_name='curvature' # 'curvature' or 'cc dist'
constraint_targets=($(seq 5 0.5 100))  # start, step, last
length_targets=($(seq 8 0.5 30))  # start, step, last
ncoils=4 # 3 or 4

for idx1 in ${!constraint_targets[@]}
do
  length_target=${length_targets[idx1]}

  for idx2 in ${!constraint_targets[@]}
  do
    constraint_target=${constraint_targets[idx2]}
    echo $length_target $constraint_target
  
    # write the submit file
    SUB="_submit_triobjective_${length_target}_${constraint_target}_ncoils_${ncoils}.sub"
    echo $SUB
    if [ -f "${SUB}" ]; then
      rm "${SUB}"
    fi
    printf '%s\n' "#!/bin/bash" >> ${SUB}
    printf '%s\n' "#SBATCH -J ${constraint_name}_${constraint_target} # Job name" >> ${SUB}
    printf '%s\n' "#SBATCH -o ./slurm_output/job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
    printf '%s\n' "#SBATCH -e ./slurm_output/job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
    printf '%s\n' "#SBATCH -N 1       # Total number of nodes requested" >> ${SUB}
    printf '%s\n' "#SBATCH -n 1       # Total number of cores requested" >> ${SUB}
    printf '%s\n' "#SBATCH --ntasks-per-node 1    # Total number of cores requested" >> ${SUB}
    printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
    printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
    printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
    #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
    #printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
    printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[29-30],g2-cpu-[97-99],g2-compute-[94-97],luxlab-cpu-02" >> ${SUB}
    printf '%s\n' "mpiexec -n 1 python3 triobjective.py ${constraint_name} ${constraint_target} ${length_target} ${ncoils}" >> ${SUB}
    
    # submit
    sbatch --requeue ${SUB}
  
  done
done
