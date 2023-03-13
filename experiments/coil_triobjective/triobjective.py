import os
from pathlib import Path
import uuid
import numpy as np
import pickle
from scipy.optimize import minimize
import sys
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves, \
    CurveLength, curves_to_vtk, MeanSquaredCurvature, CurveSurfaceDistance
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.objectives import SquaredFlux, QuadraticPenalty

"""
Run with 
  mpiexec -n 1 python3 triobjective.py ${constraint_name} ${constraint_target} ${length_target} ${ncoils} ${start_type} 
i.e.
  mpiexec -n 1 python3 triobjective.py curvature 12.0 16.0 4 cold
"""

# cmd line args
constraint_name = sys.argv[1]  # curvature or coil-coil dist
length_target_list = [float(sys.argv[3])] # float, length target 
ncoils = int(sys.argv[4]) # default is 4
start_type = sys.argv[5] # 'warm' or 'cold'
if start_type == "warm":
    constraint_target_list = np.linspace(5,50.0,100)[::-1]
else:
    constraint_target_list = [float(sys.argv[2])] # float, typically in (8,50)


# penalty options
n_penalty_solves = 3
penalty_gamma = 10
initial_penalty_weights = [0.1,1.0] # length, curvature

# solver options
maxiter = 2000
gtol = 1e-6
options = {'gtol':gtol, 'maxiter': maxiter, 'maxcor': 300} #'iprint': 5}

# coil-surface distance params
coil_surf_dist_penalty_weight = 1e6
coil_surf_dist_rhs = 0.01

# coil params
order = 5 # num fourier modes per coils
current = 1e5
# surface definition
vmec_input = '../../vmec_input_files/input.LandremanPaul2021_QA'
ntheta=nphi=32


##################################
## Done with options
##################################

objective_names = ['Quadratic flux', 'length'] + [constraint_name]

# Initialize the boundary magnetic surface:
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="half period", nphi=nphi, ntheta=ntheta)
R0 = surf.get("rc(0,0)")
R1 = R0/2

print('surf minor radius',surf.minor_radius())
print('surf area',surf.area())

# plot the surface
surf.to_vtk("surf")

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

# write vtk files
#curves = [c.curve for c in coils]
#curves_to_vtk(curves, "coils_init")

# Define the individual terms objective function:
# quadratic flux
Jflux = SquaredFlux(surf, bs)
# coil length
Jlength = [CurveLength(c) for c in base_curves]
Jlength_total = sum(Jlength)
# mean squared curvature
Jmsc = [MeanSquaredCurvature(c) for c in base_curves]
Jmsc_total = sum(Jmsc)
# coil to surface distance
Jcoil_surf_dist = CurveSurfaceDistance(base_curves,surf,coil_surf_dist_rhs)


if constraint_name == 'curvature':
    constraint = Jmsc_total
elif constraint_name == 'cc dist':
    # TODO: add coil-coil distance constraint option
    raise NotImplementedError



for li, length_target in enumerate(length_target_list):
    
    for ci, constraint_target in enumerate(constraint_target_list):
    
        print("")
        print(f"Solve {li},{ci})")
        print("length bound:",length_target)
        print("constraint bound:",constraint_target)
        print("initital Bnormal:",Jflux.J())
        print("initial length:",Jlength_total.J())
        print("initial length violation:",max(Jlength_total.J()-length_target,0.0))
        print("initial constraint:",constraint.J())
        print("initial constraint violation:",max(constraint.J()-constraint_target,0.0))
        print("initial cs-dist:",Jcoil_surf_dist.shortest_distance())
    
        # initial run params
        penalty_weights = np.array(initial_penalty_weights)
    
        # penalty solve
        for pi in range(n_penalty_solves):
        
            # penalty objective
            penalty_weights = penalty_weights * penalty_gamma
            JF = Jflux + penalty_weights[0] * QuadraticPenalty(Jlength_total,length_target, f='max')\
                + penalty_weights[1] * QuadraticPenalty(constraint,constraint_target, f='max')\
                + coil_surf_dist_penalty_weight*Jcoil_surf_dist
            def fun(dofs):
                JF.x = dofs
                return JF.J(), JF.dJ()
          
            # cold start for every new length
            if (ci == 0) and (pi == 0):
                x0 = np.copy(JF.x)
                dofs = np.copy(x0)
        
            # solve
            res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
                           options=options, tol=1e-15)
            xopt = res.x
    
            # print some stuff
            JF.x = xopt
            Fopt = np.array([Jflux.J(),Jlength_total.J(),constraint.J()])
            print(f"P{pi} Bnormal:",Jflux.J())
            print(f"P{pi} length:",Jlength_total.J())
            print(f"P{pi} length violation:",max(Jlength_total.J()-length_target,0.0))
            print(f"P{pi} constraint:",constraint.J())
            print(f"P{pi} constraint violation:",max(constraint.J()-constraint_target,0.0))
            print(f"P{pi} cs-dist:",Jcoil_surf_dist.shortest_distance())
        
        print("final Bnormal:",Jflux.J())
        print("final length:",Jlength_total.J())
        print("final length violation:",max(Jlength_total.J()-length_target,0.0))
        print("final constraint:",constraint.J())
        print("final constraint violation:",max(constraint.J()-constraint_target,0.0))
        print("final cs-dist:",Jcoil_surf_dist.shortest_distance())
    
        # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
        outputdir = f"./output/triobjective/{constraint_name}"
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        # generate a unique id
        iden = str(uuid.uuid4())
        filename_body = f"/triobjective_eps_con_length_{length_target}_{constraint_name}_{constraint_target}_{start_type}_ncoils_{ncoils}" + f"_{iden}"
        outfilename = outputdir + filename_body + ".pickle"
        outdata = {}
        outdata['xopt'] = xopt
        outdata['constraint_target'] = constraint_target
        outdata['length_target'] = length_target
        outdata['Fopt'] = Fopt
        outdata['start_type'] = start_type
        outdata['ncoils'] = ncoils
        outdata['order'] = order
        outdata['ntheta'] = ntheta
        outdata['nphi'] = nphi
        outdata['objective_names'] = objective_names
        outdata['constraint_name'] = constraint_name
        outdata['coil_surf_dist_penalty_weight'] = coil_surf_dist_penalty_weight
        outdata['coil_surf_dist_rhs'] = coil_surf_dist_rhs
        outdata['min_coil_surf_dist'] = Jcoil_surf_dist.shortest_distance()
        pickle.dump(outdata, open(outfilename,"wb"))
    
        # write vtk files
        curves = [c.curve for c in bs.coils]
        outfilename = outputdir + filename_body
        curves_to_vtk(curves, outfilename)
        
        # set starting point for next solve
        if start_type == "warm":
            JF.x = np.copy(xopt)
            dofs = np.copy(xopt)
        else:
            JF.x = np.copy(x0)
            dofs = np.copy(x0)
    
print("")
print("Done")
print("=======================================")
