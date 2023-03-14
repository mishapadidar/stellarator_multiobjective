
## Multiobjective optimization of (Aspect Ratio, Quasisymmetry)

`find_pareto_front.py` reads through the data and collects the points `X` and function
values `FX` of pareto optimal data. This should be run after `eps_con.py` to accumulate
a pickle file of pareto optimal points. These points are used for warm starting `eps_con.py`
and `predictor_corrector.py`.

We principly use the epsilon constraint method for minimizing quasisymmetry subject 
to an inequality constraint on aspect ratio. This code is in `eps_con.py`. `eps_con.py`
can be submitted on graphite via the submission script `batch_submit_eps_con.py`.
If `warm=True` then the script will warm start from the pareto optimal point with aspect
ratio nearest `aspect_target`. So you must make sure that you have run `find_pareto_front.py`
at some point before the run starts, in order to generate the pickle file with the 
pareto front.

The second method we use is a local expansion for locally exploring the pareto front
This code is in `predictor_corrector.py` and can be sumbbit to G2 via
`batch_submit_predictor_corrector.py`. This script starts its iteration from points on the 
pareto front, so `find_pareto_front.py` should be run at some point before running 
`predictor_corrector.py`. 

----------------------------
## Plotting

First run `find_pareto_front.py` and `make_pareto_plot_data.py`. This will generate the pareto 
front and make data for the pareto front plot.

Now run `select_example_points.py` to select a few pareto optimal points to make other plots for.

`plot_pareto_front.py` can then be used to plot the pareto front.

`make_paraview_data.py`, which will generate the data you need to plot the configurations in paraview.

`make_field_strength_contour_data.py` will generate data to plot the B-field contours.

`plot_field_strength_contour_data.py` plots the field strength contours.

`plot_cross_sections.py` makes the cross section plots. Relies on wout data from the selected
configurations.

