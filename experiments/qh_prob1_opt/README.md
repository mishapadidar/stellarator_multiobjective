
## Multiobjective optimization of (Aspect Ratio, Quasisymmetry)

We principly use the epsilon constraint method for minimizing quasisymmetry subject 
to an inequality constraint on aspect ratio. This code is in `eps_con.py`. `eps_con.py`
can be submitted on graphite via the submission script `batch_submit_eps_con.py`.
If `warm=True` then the script will warm start from the pareto optimal point with aspect
ratio nearest `aspect_target`. So you must make sure that you have run `find_pareto_front.py`
at some point before the run starts, in order to generate the pickle file with the 
pareto front.

The second method we use is a local expansion for locally exploring the pareto front, from
"Gradient‑based Pareto front approximation applied to turbomachinery shape optimization", 
Vasilopoulos, 2019. This code is in `predictor_corrector.py` and can be sumbbit to G2 via
`batch_submit_predictor_corrector.py`. This script starts its iteration from points on the 
pareto front, so `find_pareto_front.py` should be run at some point before running 
`predictor_corrector.py`. There is a natural ill-conditioning in this formulation,
mathematically: the hessian of the objective is rank 2 if x achieves a value of zero
in the objective. This prevents making good predictor steps. So in order to remedy this
we must ensure we set targets that are "unreachable", in that they dominate the pareto front.
We attempt to do this by using the target setting formulation from 
"Multiobjective optimal control methods for the Navier-Stokes equations using reduced order 
modeling", Peitz, 2017. The idea is to set the target by first move along the tangent to 
the pareto front, then moving along the normal vector to the pareto front, see Figure 4. 
We also rescale the quasisymmetry objective to prevent ill conditioning of the hessian.
