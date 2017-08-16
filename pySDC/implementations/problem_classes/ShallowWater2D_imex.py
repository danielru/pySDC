import numpy as np
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError
from pySDC.implementations.datatype_classes.firedrake_mesh import mesh, rhs_imex_mesh

from firedrake import Function, SpatialCoordinate, LinearVariationalProblem, LinearVariationalSolver, as_vector, cos, sin, FunctionSpace, TestFunctions, inner, div, grad, dx, NonlinearVariationalProblem, NonlinearVariationalSolver, split
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
        self.mesh = mesh
        self.Deqn = AdvectionEquation(mesh.state, mesh.state.spaces("DG"))
        self.ueqn = AdvectionEquation(mesh.state, mesh.state.spaces("HDiv"), vector_manifold=True)
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
        me.f = u0.f.copy(deepcopy=True)
        u, D = split(me.f)
        u_in, D_in = rhs.f.split()
        print("Doing solve")
        print("MINMAX depth before solve:", D_in.dat.data.min(), D_in.dat.data.max())
        u_in, D_in = split(rhs.f)

        state = self.mesh.state
        W = state.W
        w, phi = TestFunctions(W)
        f = state.fields("coriolis")
        g = state.parameters.g

        eqn = (inner(w, u)
               - factor*(
                   - f*inner(w, state.perp(u))
                   + g*div(w)*D
               )
               - inner(w, u_in)
            + phi*D
               - factor*(
                   inner(grad(phi*D), u)
                   )
               - phi*D_in
        )*dx

        params = {'snes_monitor': True,
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'schur',
                  'ksp_type': 'gmres',
                  'ksp_max_it': 100,
                  'ksp_gmres_restart': 50,
                  'pc_fieldsplit_schur_fact_type': 'FULL',
                  'pc_fieldsplit_schur_precondition': 'selfp',
                  'fieldsplit_0_ksp_type': 'preonly',
                  'fieldsplit_0_pc_type': 'bjacobi',
                  'fieldsplit_0_sub_pc_type': 'ilu',
                  'fieldsplit_1_ksp_type': 'preonly',
                  'fieldsplit_1_pc_type': 'gamg',
                  'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                  'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                  'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                  'fieldsplit_1_mg_levels_ksp_max_it': 1,
                  'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                  'fieldsplit_1_mg_levels_sub_pc_type': 'ilu'}

        hyb_params = {
            'snes_max_it': 50,
            'snes_atol': 1e-6,
            'ksp_type': 'preonly',
            'mat_type': 'matfree',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'cg',
                              'pc_type': 'gamg',
                              'ksp_rtol': 1e-8,
                              'mg_levels': {'ksp_type': 'chebyshev',
                                            'ksp_max_it': 2,
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type': 'ilu'},
                              # Broken residual construction
                              'hdiv_residual': {'ksp_type': 'cg',
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu',
                                                'ksp_rtol': 1e-8},
                              # Projection step
                              'hdiv_projection': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-8}}
        }
        prob = NonlinearVariationalProblem(eqn, me.f)
        solver = NonlinearVariationalSolver(prob, solver_parameters=params)
        solver.solve()
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
        print("MINMAX depth before expl:", Dn.dat.data.min(), Dn.dat.data.max())

        fexpl = self.dtype_u(self.init)
        uout, Dout = fexpl.f.split()
        Dout.assign(0.0)
        uout.assign(as_vector([0., 0.]))

        self.Deqn.ubar.project(un)
        Dlhs = self.Deqn.mass_term(self.Deqn.trial)
        Drhs = self.Deqn.advection_term(Dn)
        Dprob = LinearVariationalProblem(Dlhs, Drhs, Dout)
        Dsolver = LinearVariationalSolver(Dprob, solver_parameters={'ksp_monitor_true_residual': True})
        Dsolver.solve()

        self.ueqn.ubar.project(un)
        ulhs = self.ueqn.mass_term(self.ueqn.trial)
        urhs = self.ueqn.advection_term(un)
        uprob = LinearVariationalProblem(ulhs, urhs, uout)
        usolver = LinearVariationalSolver(uprob, solver_parameters={'ksp_monitor_true_residual': True})
        usolver.solve()
        print("MINMAX depth after expl:", Dout.dat.data.min(), Dout.dat.data.max())

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
        print("MINMAX depth impl:", Dn.dat.data.min(), Dn.dat.data.max())

        fimpl = self.dtype_u(self.init)

        uout, Dout = fimpl.f.split()
        Dout.assign(0.0)

        Dlhs = self.Deqn.mass_term(self.Deqn.trial)
        Drhs = self.forcing.divu_term(u.f)
        Dprob = LinearVariationalProblem(Dlhs, Drhs, Dout)
        Dsolver = LinearVariationalSolver(Dprob)
        Dsolver.solve()

        ulhs = self.ueqn.mass_term(self.ueqn.trial)
        urhs = self.forcing.pressure_gradient_term(u.f) + self.forcing.coriolis_term(u.f)
        uprob = LinearVariationalProblem(ulhs, urhs, uout)
        usolver = LinearVariationalSolver(uprob)
        usolver.solve()
        print("MINMAX depth after impl:", Dout.dat.data.min(), Dout.dat.data.max())

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
        state = mesh.state
        mymesh = mesh.state.mesh
        R = mesh.R
        u0 = state.fields("u")
        D0 = state.fields("D")
        omega = 7.848e-6  # note lower-case, not the same as Omega
        K = 7.848e-6
        g = state.parameters.g
        Omega = state.parameters.Omega
        H = state.parameters.H
        x = SpatialCoordinate(mymesh)

        theta, lamda = latlon_coords(mymesh)

        u_zonal = R*omega*cos(theta) + R*K*(cos(theta)**3)*(4*sin(theta)**2 - cos(theta)**2)*cos(4*lamda)
        u_merid = -R*K*4*(cos(theta)**3)*sin(theta)*sin(4*lamda)

        uexpr = sphere_to_cartesian(mymesh, u_zonal, u_merid)

        def Atheta(theta):
            return 0.5*omega*(2*Omega + omega)*cos(theta)**2 + 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 + 26 - 32/(cos(theta)**2))

        def Btheta(theta):
            return (2*(Omega + omega)*K/30)*(cos(theta)**4)*(26 - 25*cos(theta)**2)

        def Ctheta(theta):
            return 0.25*(K**2)*(cos(theta)**8)*(5*cos(theta)**2 - 6)

        Dexpr = H + (R**2)*(Atheta(theta) + Btheta(theta)*cos(4*lamda) + Ctheta(theta)*cos(8*lamda))/g

        # Coriolis
        fexpr = 2*Omega*x[2]/R
        V = FunctionSpace(mymesh, "CG", 1)
        f = state.fields("coriolis", V)
        f.interpolate(fexpr)  # Coriolis frequency (1/s)

        u0.project(uexpr, form_compiler_parameters={'quadrature_degree': 8})
        D0.interpolate(Dexpr)

        state.initialise([('u', u0), ('D', D0)])

        me = self.dtype_u(self.init)
        me.f = state.xn
        return me
