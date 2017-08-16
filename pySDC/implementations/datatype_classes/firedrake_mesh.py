from firedrake import *
from gusto import *

import numpy as np
import copy as cp
from pySDC.core.Errors import DataError

class mesh(object):
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """
  
    # BEWARE: THIS MEANS YOU WILL HAVE TO KEEP THIS IN SYNC WITH WHATEVER YOU GIVE THE CODE IN YOUR PYSDC PARAMETERS
    # !!!111elfelfCAPSLOCK
    R = 6371220
    ref_level = 3
    mymesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_level, degree=3)
    # Create GUSTO mesh and state
    ref_level = 3
    dirname = "sw_W2_ref%s" % (ref_level)
    R = 6371220. # in metres
    day = 86400.
    x = SpatialCoordinate(mymesh)
    global_normal = x
    mymesh.init_cell_orientations(x)
    parameters = ShallowWaterParameters()
    output = OutputParameters(dirname=dirname, dumplist_latlon=['D', 'D_error'], steady_state_error_fields=['D', 'u'])
    fieldlist = ['u', 'D']
    state = State(mymesh, horizontal_degree=1,
                           family="BDM",
                           output=output,
                           parameters=parameters,
                           diagnostics=diagnostics,
                           fieldlist=fieldlist)       


    def __init__(self, init=None, val=None):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another mesh, do a deepcopy (init by copy)
        if isinstance(init, mesh):
            # deepcopy of f which should be a firedrake function
            self.f = init.f.copy(deepcopy=True)  

        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init, tuple) or isinstance(init, int):

            ### FIXME: for now, we assume that the mesh and function space is always the same and hard-code it here. Bad code! ###
            assert isinstance(init, int), NotImplementedError("Cannot yet use firedrake mesh init routine with tupel sized data")
            self.f = Function(mesh.state.W)
            #self.f.interpolate(Expression("x[0] = a", a=val))

        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (mesh.mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: sum of caller and other values (self+other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me   = mesh(self)   
            me.f += other.f
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other (mesh.mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            mesh.mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = mesh(self)
            me.f -= other.f
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            mesh.mesh: copy of original values scaled by factor
        """

        # always create new mesh, since otherwise c = a + b changes a as well!
        me = mesh(self)
        me.f *= other
        return me

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: absolute maximum of all mesh values
        """

        # return maximum
        u, D, = self.f.split()
        return np.linalg.norm(u.vector(), np.inf) + np.linalg.norm(D.vector(), np.inf)

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.mesh: component multiplied by the matrix A
        """
        raise NotImplementedError("apply_mat is not implemented for firedrake_mesh")

class rhs_imex_mesh(object):
    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (mesh.mesh): implicit part
        expl (mesh.mesh): explicit part
    """

    def __init__(self, init):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = mesh(init.impl)
            self.expl = mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, tuple) or isinstance(init, int):
            self.impl = mesh(init)
            self.expl = mesh(init)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (mesh.rhs_imex_mesh): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values - other.impl.values
            me.expl.values = self.expl.values - other.expl.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (mesh.rhs_imex_mesh): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            mesh.rhs_imex_mesh: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values + other.impl.values
            me.expl.values = self.expl.values + other.expl.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
             mesh.rhs_imex_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = other * self.impl.values
            me.expl.values = other * self.expl.values
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def apply_mat(self, A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh.rhs_imex_mesh: each component multiplied by the matrix A
        """

        if not A.shape[1] == self.impl.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.impl))
        if not A.shape[1] == self.expl.values.shape[0]:
            raise DataError("ERROR: cannot apply operator %s to %s" % (A, self.expl))

        me = rhs_imex_mesh(A.shape[1])
        me.impl.values = A.dot(self.impl.values)
        me.expl.values = A.dot(self.expl.values)

        return me
