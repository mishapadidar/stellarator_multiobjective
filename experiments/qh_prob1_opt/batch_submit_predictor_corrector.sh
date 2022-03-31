
#ASPECTS=('5.075' '8.0' '8.3' '4.95') # for moving left
ASPECTS=('4.4' '6.2' '8.7') # for moving right
vmec="high" # use high for aspect < 4.5, and when singular jacobians appear.
maxmode=5
direction="right"
NODES=1
CORES=12
for idx in ${!ASPECTS[@]}
do
  aspect=${ASPECTS[idx]}

  # make a dir
  mkdir "_batch_pc_${aspect}"

  # copy the penalty method
  cp "./predictor_corrector.py" "_batch_pc_${aspect}/predictor_corrector.py"

  # write the run file
  RUN="_batch_pc_${aspect}/run.sh"
  if [ ! -f "${RUN}" ]; then
    printf '%s\n' "sbatch --requeue submit.sub" >> ${RUN}
    chmod +x ${RUN}
  fi

  # write the submit file
  SUB="_batch_pc_${aspect}/submit.sub"
  if [ -f "${SUB}" ]; then
    rm "${SUB}"
  fi
  printf '%s\n' "#!/bin/bash" >> ${SUB}
  printf '%s\n' "#SBATCH -J pc_${aspect} # Job name" >> ${SUB}
  printf '%s\n' "#SBATCH -o ./job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -e ./job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -N ${NODES}       # Total number of nodes requested" >> ${SUB}
  printf '%s\n' "#SBATCH -n ${CORES}       # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --ntasks-per-node ${CORES}    # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=2000   # Memory required per allocated CPU" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99],g2-compute-[94-97]" >> ${SUB}
  printf '%s\n' "#SBATCH --exclusive" >> ${SUB}
  printf '%s\n' "mpiexec -n $[$NODES*$CORES] python3 predictor_corrector.py ${aspect} ${vmec} ${maxmode} ${direction}" >> ${SUB}
  
  ## submit
  cd "./_batch_pc_${aspect}"
  "./run.sh"
  cd ..

  ## sleep to prevent job clashes
  #sleep 3
done

