## Bi-objective optimization of stellarator coil configurations

We consider the trade-off between coil length and quadratic flux.

Some script use different versions of SIMSOPT. Each script should say.

Generate the main data by running `biobjective.py`. You can submit jobs
which run the epsilon constraint method by using `batch_submit_biobjective.sh`.
By using the `warm_mode=True` option you can warm start runs. This requires 
running the script in serial using the `submit.sub` file to submit to SLURM.

After generating data run `plot_pareto.py` to make a plot of the pareto front.
It will also copy the data for the pareto optimal points to a new directory,
which is used in the following experiments.

Generate data for contour plots of the coil field strength on the s=1 surface using
`make_field_strength_contour_data.py`. Plot it with `plot_field_strength_contour.py`.

Build QFM surfaces and estimate the quasi-symmetry level of the coil field using
`make_qfm_surface_data.py`. plot the data with `plot_qfm.py`
