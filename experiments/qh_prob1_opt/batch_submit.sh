
ASPECTS=('3.0' '4.0' '5.0' '6.0' '7.0')
NODES=10
for aspect in "${ASPECTS[@]}"
do
  # make a dir
  mkdir "_batch_aspect_${aspect}"

  # copy the penalty method
  cp "./penalty_method.py" "_batch_aspect_${aspect}/penalty_method.py"

  # write the run file
  RUN="_batch_aspect_${aspect}/run.sh"
  printf '%s\n' "sbatch --requeue submit.sub" >> ${RUN}
  chmod +x ${RUN}

  # write the submit file
  SUB="_batch_aspect_${aspect}/submit.sub"
  printf '%s\n' "#!/bin/bash" >> ${SUB}
  printf '%s\n' "#SBATCH -J aspect_${aspect} # Job name" >> ${SUB}
  printf '%s\n' "#SBATCH -o ./job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -e ./job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -N ${NODES}       # Total number of nodes requested" >> ${SUB}
  printf '%s\n' "#SBATCH -n ${NODES}       # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
  printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99]" >> ${SUB}
  printf '%s\n' "mpiexec -n ${NODES} python3 penalty_method.py ${aspect} ../data" >> ${SUB}
  
  ## submit
  cd "./_batch_aspect_${aspect}"
  "./run.sh"
  cd ..
done

