import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

#from projects.FastWaveSlowWave.HookClass_acoustic import dump_energy
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.implementations.datatype_classes.firedrake_mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
#from pySDC.implementations.problem_classes.acoustic_helpers.standard_integrators import bdf2, dirk, trapezoidal, rk_imex

def compute_and_plot_solutions():
    """
    Routine to compute and plot the solutions of SDC(2), IMEX, BDF-2 and RK for a multiscale problem
    """

    num_procs = 1

    t0 = 0.0
    Tend = 3.0
    nsteps = 1  # 154 is value in Vater et al.
    dt = Tend / float(nsteps)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = dt

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] = 2

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['cadv']      = 0.05
    problem_params['cs']        = 1.0
    problem_params['nvars']     = [(16)]
    problem_params['order_adv'] = 5
    problem_params['waveno']    = 5

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLegendre
    sweeper_params['num_nodes'] = 2

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    #controller_params['hook_class'] = dump_energy

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = shallowwater_imex
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params
    description['level_params'] = level_params

    # instantiate the controller
    controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                             description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

if __name__ == "__main__":
    compute_and_plot_solutions()
