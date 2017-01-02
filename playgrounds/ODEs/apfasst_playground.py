import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.complex_mesh import mesh
from pySDC.implementations.problem_classes.AdvectionDiffusionSingleMode import advection_diffusion_spectral
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.helpers.stats_helper import filter_stats, sort_stats, get_list_of_types
from pySDC.implementations.transfer_classes.TransferAPfasst import apfasst_transfer

#from playgrounds.ODEs.trajectory_HookClass import trajectories


def main():
    """
    Advection-diffusion equation for a single mode
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-8
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5, 2]
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['kappa'] = 1
    problem_params['U']     = 1
    problem_params['nu']    = 0.0

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['U'] = problem_params['U'] ## Can may be access problem_params['U'] directly in transfer class?
    space_transfer_params['kappa'] = problem_params['kappa'] ## see above

    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 15

    # initialize controller parameters
    controller_params = dict()
    #controller_params['hook_class'] = trajectories
    controller_params['logger_level'] = 30
    controller_params['log_to_file'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = advection_diffusion_spectral
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = apfasst_transfer  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # instantiate the controller
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0   = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    #print uinit.values[0]
    
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by time
    iter_counts = sort_stats(filtered_stats, sortby='time')
    print iter_counts
    for item in iter_counts:
      out = 'Number of iterations at time %4.2f: %2i' % item
      #f.write(out + '\n')
      #print(out)
    
    # compute exact solution and compare
    uex = P.u_exact(Tend)
    #print uex.values
    #print uend.values
    err = abs(uex - uend)

    print('\nError: %8.6e' % err)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
