from __future__ import division

import numpy as np

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class advection_diffusion_spectral(ptype):
    """
    Example implementing the initial value problem coming from the advection-diffusion equation for a single mode.
    Fourier transformation of the advection-diffusion equation u_t + U u_x = nu u_xx
    yields the following ODE for every mode:
    
      U_t = -( U*i*k + nu*k^2 ) U
      
    with solution
    
      U(t) = exp(-U*i*k)*exp(-nu*k^2*t)
      
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['kappa', 'U', 'nu']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing dtype_u and dtype_f, plus setting number of elements to 1
        super(advection_diffusion_spectral, self).__init__(1, dtype_u, dtype_f, problem_params)
        
    def u_exact(self, t):
        """
        Routine for the exact solution

        Args:
            t (float): current time
        Returns:
            dtype_u: mesh type containing the exact solution
        """

        me = self.dtype_u(self.init)
        me.values[0] = np.exp(-self.params.nu*self.params.kappa**2*t)*np.exp(-self.params.U*1j*t)
        return me

    def eval_f(self, u, t):
        """
        Routine to compute the RHS for both components simultaneously

        Args:
            u (dtype_u): the current values
            t (float): current time (not used here)
        Returns:
            RHS, 2 components
        """

        x1          = u.values[0]
        f           = self.dtype_f(self.init)
        f.values[0] = -(self.params.U*1j*self.params.kappa + self.params.nu*self.params.kappa**2)*x1
        return f

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            dt (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution u
        """

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = rhs.values[0]
        u.values[0] = x1/(1.0 + dt*(self.params.U*1j*self.params.kappa + self.params.nu*self.params.kappa**2))

        return u