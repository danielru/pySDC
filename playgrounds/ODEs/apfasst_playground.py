import matplotlib.pyplot as plt

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.complex_mesh import mesh
from pySDC.implementations.problem_classes.AdvectionDiffusionSingleMode import advection_diffusion_spectral
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.helpers.stats_helper import filter_stats, sort_stats, get_list_of_types
from TransferAPfasst import apfasst_transfer

#from playgrounds.ODEs.trajectory_HookClass import trajectories
import numpy as np

def main():
    """
    Advection-diffusion equation for a single mode
    """
    
    
    # set time parameters
    t0   = 0.0
    Tend = 0.5
    
    # set number of processors
    nproc = 1
    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-14
    level_params['dt']     = (Tend - t0)/float(nproc)

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['QI'] = 'IE'

    # initialize problem parameters
    problem_params = dict()

    ### PFASST
    sweeper_params['num_nodes'] = [2, 2]
    problem_params['kappa'] = [1.0, 1.0]
    problem_params['U']     = [1.0, 0.0] # advection is being taken care of by transfer operators
    problem_params['nu']    = [0.0, 0.0]
    problem_params['dx']    = [0.0, 0.0]


    ### SDC
    #sweeper_params['num_nodes'] = [2]
    #problem_params['kappa'] = [1.0]
    #problem_params['U']     = [1.0] # advection is being taken care of by transfer operators
    #problem_params['nu']    = [0.0]
    #problem_params['dx']    = [0.0]

    # initialize space transfer parameters
    space_transfer_params = dict() ### Won't be used
    space_transfer_params['rorder'] = 1
    space_transfer_params['iorder'] = 1
    
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

    apfasst_transfer_params = dict()

    description['base_transfer_class']  = apfasst_transfer
    description['base_transfer_params'] = apfasst_transfer_params
    
    description['space_transfer_class']  = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # instantiate the controller
    controller = allinclusive_classic_nonMPI(num_procs=nproc, controller_params=controller_params, description=description)


    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    #print uinit.values[0]
    
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, time=t0, type='residual_post_iteration')
    
    # sort and convert stats to list, sorted by iteration numbers
    residuals = sort_stats(filtered_stats, sortby='iter')

    for item in residuals:
        out = 'Residual in iteration %2i: %8.4e' % item
        #f.write(out + '\n')
        print(out)


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