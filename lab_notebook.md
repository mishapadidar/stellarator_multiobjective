## Lab Notebook

### ToDo
- Write a Nelder-Mead with Tchebycheff solver
- Rerun NSGA-II without bound constraints
- plot the pareto front
- select a few points to make VMEC plots
- Make bounding boxes around different starting points. Are these points within our original bounds?
  > `examples/2_Intermediate/inputs/input.nfp2_QA`
  > circular axisymmetric start
  > RBC(0,0) =   1.0E+00     ZBS(0,0) =   0.0000E+00
  >   RBC(0,1) =   2.0E-01     ZBS(0,1) =   2.0E-01
  > 
  > helical twist on axis
  > RBC( 0,  0) = 10.0   ZBS( 0, 0) = 0.0
  > RBC( 1,  0) =  2.0   ZBS( 1, 0) = 2.0
  > RBC( 0,  1) =  1.7   ZBS( 0, 1) = 1.7
  > make sure to divide these numbers by 10!
- Reoptimize in the extended domain.
- Analyze QHProb1 pareto front
  - look at convexity vs non-convexity.
  - choose a few points of interest.
  - Verify pareto optimality of these points.
  - locally expand pareto front around non-dominated points.
  - Look at sensitivity to surface labels.
- Look at other variations of the problem
  - minimize QS with aspect ratio equality constraint.
  - do multiobjective opt with [QS, iota] and aspect equality constraint.


### Completed
  - [x] Talk to Matt about:
    - [x] turning off VMEC output.
    - [x] vmec discretization.
    - [x] built in finite difference for QS and aspect.
    - [x] parallelism: How can we concurrently evaluate the objective?
    - [x] sim failure feasible region ansatz.
  - [x] Set up vmec resolution parameters in the QH prob. 
     > Denote as `vmec.indata.mpol` and `vmec.indata.ntor`. Keep this higher than 
       the max Fourier modes of the boundary.
  - [x] write a finite difference jacobian.
  - [x] test finite difference jacobian.
  - [x] Write a parallel jacobian computation using working groups.
  - [x] Run Matt's QS test.
     > Seemed succesful. Here is the print out:
     > `xtol` termination condition is satisfied. 
     > Function evaluations 30, initial cost 1.5177e-01, final cost 1.7038e-03, first-order optimality 6.35e+01.
     > Time to solve:  303.22242522239685
     > Final Obj:  0.0034076812956755353
     > Final Total QS:  0.0034056233511294337
  - [x] Performing Timing Test
     > Using more mpi resources while keeping the number of worker groups fixed does not speed up compuation.
     > In fact I found that it was fastest to use a single mpi rank per worker group. 
     > Thus it is best to use only 1 rank per MPIPartition, and leverage concurrent computations instead.
     >
     > At max_mode = 1 (8 dofs) the computational time is 1.5sec. 
     > At max_mode = 2 (24 dofs) the computational time is 3.3sec. 
     > At max_mode = 3 (48 dofs) the computational time is 4.1sec. 
     >
     > VMEC print statements take 0.25sec - 0.35sec.
  - [x] Upgrade concurrent evals and jacobian in qh\_prob1.py so that arbitrary number of
        points can be evaluated.
  - [x] Make 1d-plots to show finite difference fidelity, multi-modality, and simulation failures.
       > Plot shows no noise even up to 1e-9 step size. This shows pretty amazing precision in VMEC solves at the
       > current resolution (by default set to 5 = 2dofs + 3).
       > We do see noise in steps of size 1e-10 along all directions at x0 for the low fidelity (original) input file.
       > Functions look like convex quadratics around qh_prob1's x0.
       > 
       > See figure `1dplot_low_res_noise.png` from data `1dplot_data_close_up.pickle`.
       >
       > Simulation failures may not be connected in space. Some directions fail, then stop failing, then fail again 
       > along a 1d line segment, which implies non-convexity of the feasible region.
       > Lastly, bounds for each variable differ greatly.
  - [x] Write a script to sample the feasible region.
  - [x] Consdier three approximations to the feasible region: safe approximation, outer approximation, best approximation.
       > safe approximation is useful to optimization methods as it gaurantees to avoids failures. Typically safe approximation
       > should use a non-convex feasible region approximant, so that it consumes as much of the feasible region as possible. 
       > The 2d plots show that the safe approximation would be quite small for our problem.
       >
       > The best approximation is typically a nonlinear classification methods, such as an SVM. It can be used as a penalty in 
       > objective functions to push methods away from the boundary and improve the use of the computational budget. Nonlinear,
       > nonconvex regions, such as the best approximation, are difficult to sample uniformly.
       >
       > The safe approximation can be any shape so long as it contains the feasible region. Bound constrained safe
       > approximations are useful for sampling from the feasible region, rescaling input parameters, and bounding 
       > global and multi-objective optimization methods.
  - [x] SIMSOPT does not seem to run on some of the G2 nodes, specifically some (and perhaps all, though not tested) of 
        g2-cpu-[06-11,97-98]. We can get around running on these by including the following in the slurm submit script 
        `SBATCH --exclude=g2-cpu-[06-11,97-98]`
  - [x] Look in to SurfaceRZPseudoSpectral coordinates
       > `simsopt` vesion `0.7.0` does not have this feature. Need to upgrade to get access.
  - [x] make 2d slices of feasible region.
  - [x] Write script to build data-based bound constraints.
  - [x] Discuss the catastrophic failure with Matt. 
       > The catastrophic failure (which kills the python run) is due to a self-intersecting surface.
       > VMEC no longer fails catastrophically (no longer kills the python run) when evaluating a select point after
       > we increase the VMEC `mpol` and `ntor` parameters to `6` or higher from `5`. There is no guarantee that this holds
       > at all points which catastrophically fail. 
  - [x] Look at the effect of the VMEC resolution parameters on simulation failures. Regenerate the 1d and 2d plots with 
        high res VMEC parameters.
      > Increasing the the number of VMEC iterations from 2000 to 5000 and increasing DELT to 0.5 from 0.9 (a step size param)
        removes a significant amount of the VMEC failures. The 
      > 2d plots went from looking like swiss cheese to a solid piece of cheese, i.e. the feasible region is no longer laden 
      > with holes.
      >
      > See figure `2dplot_data_6_15_high_res.png` with data from `2dplot_data_6_15_high_res.pickle`.
  - [x] Run 2D plot with high res parameters except `DELT=0.9` to see effect of `DELT`
      > Even with all other resolution parameters at the higher setting, setting `DELT=0.9` (the original setting) leads to 
      > simulation failures, making the 2d plots once again look like swiss cheese. I recommend using the high res
      > setting `DELT=0.5`
      > 
      > See the figure `2dplot_data_6_15_high_res_DELT09.png` with data from `2dplot_data_6_15_high_res_DELT09.pickle`.
  - [x] Run a timing test with the new high resolution.
      > DELT has a significant effect on runtime, adding a few seconds when increased from 0.9 to 0.5. 
      > The DELT parameter does have a siginificant effect on the occurence of failures, DELT=0.5 being superior to DELT=0.9. 
      > Using DELT=0.5 also
      > helps prevent some, but not all, of the catastrophic failures. 
      > Increasing mpol,ntor by one seems to add about a second to the runtime.
      > Increasing the VMEC iterations to 5000 adds a good amount of time as well.
      >
      > With 5000 VMEC iterations, DELT=0.5, and npol=mtor=7 evaluations take 8.5-20sec, depending on the machine! Average
      > is about 15sec evaluations.
  - [x] Write a 'safe evaluation' wrapper.
  - [x] Implement safe evaluations in the data based bounds.
  - [x] Compute the data-based bounds
    - [x] Compute bounds on the objectives
         > Scaling varies greatly amongst inputs, as expected
    - [x] Compute bounds on the inputs
         > Roughly, QS goes from 0 to 1000 and aspect goes from 0 to 0.5 or so over our samples.
  - [x] choose a method of solving the unrelaxable bound constrained MOO problem.
      > Bi-objective MOO:
      > With Bi-objective we can use a discrete set of
      > reference points within a Lp or Tchebycheff approach.
      > First need to (globally) minimize each objective to bound the pareto front.
      > Use multistart (fault tolerant)GD or direct search.
      >
      > DFO MOO solvers:
      > NSGA-II, DMS, MultiMADS, ParEGO
  - [x] Plot the data in F-space to see the relative scale
      > Roughly, QS goes from 0 to 1000 and aspect goes from 0 to 0.5 or so over our samples.
  - [x] Write an `is_pareto_efficient` function.
  - [x] Set up multiobjective optimization of QHProb1
    - [x] Use NSGA-II
    - [x] Rescale objectives and inputs before optimization.
    - [x] save all evaluation data.

  
