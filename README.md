## Multi-Objective Optimization of Stellarator Plasma Boundaries

### ToDO
- Untested features and problems
  - finite difference feature
  - vmec resolution parameters
  - Matt's QS test
  - Concurrent evaluation test
- Exploratory Analysis
  - Switch to SurfaceRZPseudoSpectral coordinates
  - time evaluations and finite difference.
  - determine smart use of parallel resources (concurrent vs joint evals)
  - make 1d plots to show finite difference fidelity, 
    multi-modality, and simulation failures
  - Select VMEC discretization based on fidelity
  - make 2d slices of feasible region for SIMSOPT meeting.
- Constraints
  - build data-based bound constraints.
  - rescale the space according to the bounds.
- choose a method of solving the unrelaxable bound constrained MOO problem.
  - MOO solvers
    - NSGA-II
  - DFO:
    - POUNDERS on linearized sum of squares
    - Nelder-Mead/COBYLA/MADS on Tchebycheff/Lp problems
    - ParEGO etc...
  - Derivative Based:
    - (Sub)-GD (BFGS) with fault tolerant linesearch on scalarization.
- optimize and save all evaluation data.
- plot the pareto front.
- locally expand pareto front around non-dominated points.
- look at convexity vs non-convexity.
- choose a few points of interest.
- Look at sensitivity to surface labels.
