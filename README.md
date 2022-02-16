## Stellarator Multi-Objective Optimization

### Optimization questions and problems
- how should we deal with simulation failures?
- What algorithm should we use to solve the problem
- should we treat all QS surface objectives separately?
- how should we leverage parallelism?

### ToDO
- Talk to Matt about 
  - turning off VMEC output.
  - built in finite difference for QS and aspect.
- time evaluations and finite difference.
- select parallelism.
- make 1d plots to show finite difference fidelity, 
  multi-modality, and simulation failures
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
