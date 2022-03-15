
#ASPECTS=('3.0')
#WARM=('../data/data_aspect_3.0_20220312174044.pickle')
#ASPECTS=('4.0')
#WARM=('../data/data_aspect_4.0_20220313201431.pickle')
#ASPECTS=('5.0')
#WARM=('../data/data_aspect_5.0_697539.pickle')
#ASPECTS=('6.0')
#WARM=('../data/data_aspect_6.0_672352.pickle')
#ASPECTS=('7.0')
#WARM=('../data/data_aspect_7.0_20220312160438.pickle')
ASPECTS=('8.0')
WARM=('../data/data_aspect_8.0_20220313035614.pickle')
NODES=6
for idx in ${!ASPECTS[@]}
do
  aspect=${ASPECTS[idx]}
  warm=${WARM[idx]}
  # make a dir
  mkdir "_batch_aspect_${aspect}"

  # copy the penalty method
  cp "./penalty_method.py" "_batch_aspect_${aspect}/penalty_method.py"

  # write the run file
  RUN="_batch_aspect_${aspect}/run.sh"
  if [ ! -f "${RUN}" ]; then
    printf '%s\n' "sbatch --requeue submit.sub" >> ${RUN}
    chmod +x ${RUN}
  fi

  # write the submit file
  SUB="_batch_aspect_${aspect}/submit.sub"
  if [ -f "${SUB}" ]; then
    rm "${SUB}"
  fi
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
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99],g2-compute-[94-97]" >> ${SUB}
  printf '%s\n' "#SBATCH --exclusive" >> ${SUB}
  printf '%s\n' "mpiexec -n ${NODES} python3 penalty_method.py ${aspect} ../data ${warm}" >> ${SUB}
  
  ## submit
  cd "./_batch_aspect_${aspect}"
  "./run.sh"
  cd ..
done

