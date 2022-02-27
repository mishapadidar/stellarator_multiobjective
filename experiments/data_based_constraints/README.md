
To generate the data based bound constraints follow these steps.
1. Run `compute_bounds.py` in parallel on slurm to densely sample the interior of the
   feasible region.
2. Once complete, run `compute_bounds_safely.py`. This script runs serially but you can run
   multiple separate instances simultaneously to generate more total samples safely.
   This script will warm start the bounds by reading over all past data generated.
3. After you are satisified with the data generated run the script `combine_bounds.py` to 
   drop a pickle file with the data based bound constraints.
