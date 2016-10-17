import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA
from __future__ import division

from pySDC.Problem import ptype

class vortex2d(ptype):

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        if not 'nu' in problem_params:
            problem_params['nu'] = 1
        if not 'rho' in problem_params:
            problem_params['rho'] = 50
        if not 'delta' in problem_params:
            problem_params['delta'] = 0.05

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        assert 'nvars' in problem_params, 'ERROR: need number of nvars for the problem class'
        assert len(problem_params['nvars']) == 2, "ERROR, this is a 2d example, got %s" %problem_params['nvars']
        assert problem_params['nvars'][0] == problem_params['nvars'][1], "ERROR: need a square domain, got %s" %problem_params['nvars']
        assert problem_params['nvars'][0] % 2 == 0, 'ERROR: the setup requires nvars = 2^p per dimension'


        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(vortex2d, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                       params=problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1 / self.params.nvars
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx)

    def __get_A(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil,[-1,0,1],shape=(N,N))
        A *= nu / (dx**2)
        return A.tocsc()

    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f

    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        xvalues = np.array([(i+1)*self.dx for i in range(self.params.nvars)])
        fexpl = self.dtype_u(self.init)
        fexpl.values = -np.sin(np.pi*self.params.freq*xvalues)*(np.sin(t)-self.params.nu*(np.pi*self.params.freq)**2*np.cos(t))
        return fexpl

    def __eval_fimpl(self,u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl = self.dtype_u(self.init)
        fimpl.values = self.A.dot(u.values)
        return fimpl

    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i+1)*self.dx for i in range(self.params.nvars)])
        me.values = np.sin(np.pi*self.params.freq*xvalues)*np.cos(t)
        return me

