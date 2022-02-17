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
  - build three approximations to the VMEC feasible region
    - safe approximation, outer approximation, best approximation.
    - Have Matt take a look at points which evaluate but are 
      outside of the safe approximation
- choose a method of solving the unrelaxable bound constrained MOO problem.
  - Bi-objective MOO
    - With Bi-objective we can use a discrete set of
      reference points within a Lp or Tchebycheff approach.
      First need to (globally) minimize each objective to bound the pareto front.
      Use multistart (fault tolerant)GD or direct search.
  - DFO MOO solvers
    - NSGA-II, DMS, MultiMADS
- optimize and save all evaluation data.
- plot the pareto front.
- locally expand pareto front around non-dominated points.
- look at convexity vs non-convexity.
- choose a few points of interest.
- Look at sensitivity to surface labels.
