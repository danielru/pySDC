import numpy as np
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError
from pySDC.implementations.datatype_classes.firedrake_mesh import mesh, rhs_imex_mesh

from firedrake import Function, SpatialCoordinate, LinearVariationalProblem, LinearVariationalSolver, Expression, as_vector
from gusto import *

# noinspection PyUnusedLocal
class shallowwater_imex(ptype):
    """Example implementing the Williamson 2 shallow water test case on
    the sphere

    Attributes:
        mesh (numpy.ndarray): 1d mesh

    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data
            dtype_f: mesh data with two components
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'cs', 'cadv', 'order_adv', 'waveno']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(shallowwater_imex, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        #ueqn = VectorInvariant(state, u0.function_space())
        #Deqn = AdvectionEquation(state, D0.function_space(), equation_form="continuity")
        self.Deqn = AdvectionEquation(mesh.state, mesh.state.spaces("DG"))
        self.forcing = ShallowWaterForcing(mesh.state)

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (M-dt*f_impl)(u) = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        me.f = rhs.f.copy(deepcopy=True)
        return me

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u (dtype_u): current values (not used here)
            t (float): current time

        Returns:
            explicit part of RHS
        """

        un, Dn = u.f.split()
        print(Dn.dat.data.min(), Dn.dat.data.max())

        fexpl = self.dtype_u(self.init)
        uout, Dout = fexpl.f.split()
        Dout.assign(0.0)

        self.Deqn.ubar.project(un)
        lhs = self.Deqn.mass_term(self.Deqn.trial)
        rhs = self.Deqn.advection_term(Dn)
        prob = LinearVariationalProblem(lhs, rhs, Dout)
        solver = LinearVariationalSolver(prob)
        solver.solve()

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time (not used here)

        Returns:
            implicit part of RHS
        """

        un, Dn = u.f.split()
        print(Dn.dat.data.min(), Dn.dat.data.max())

        fimpl = self.dtype_u(self.init)
        uout, Dout = fimpl.f.split()
        Dout.assign(0.0)

        lhs = self.Deqn.mass_term(self.Deqn.trial)
        rhs = self.forcing.divu_term(u.f)
        prob = LinearVariationalProblem(lhs, rhs, Dout)
        solver = LinearVariationalSolver(prob)
        solver.solve()

        return fimpl

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """


        # interpolate initial conditions
        u0 = mesh.state.fields("u")
        D0 = mesh.state.fields("D")
        x = SpatialCoordinate(mesh.state.mesh)
        u_max = 2*np.pi*mesh.R/(12*mesh.day)  # Maximum amplitude of the zonal wind (m/s)
        uexpr = as_vector([-u_max*x[1]/mesh.R, u_max*x[0]/mesh.R, 0.0])
        Omega = mesh.state.parameters.Omega
        g = mesh.state.parameters.g
        Dexpr = Expression("R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0)) < rc ? (h0/2.0)*(1 + cos(pi*R*acos(fmin(((x[0]*x0 + x[1]*x1 + x[2]*x2)/(R*R)), 1.0))/rc)) : 0.0", R=mesh.R, rc=mesh.R/3., h0=1000., x0=0.0, x1=-mesh.R, x2=0.0)

        u0.project(uexpr)
        D0.interpolate(Dexpr)

        mesh.state.initialise([('u', u0), ('D', D0)])

        me = self.dtype_u(self.init)
        me.f = mesh.state.xn
        return me
