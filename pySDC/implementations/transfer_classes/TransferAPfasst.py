from __future__ import division

from pySDC.implementations.datatype_classes.complex_mesh import mesh #, rhs_imex_mesh
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.core.Errors import TransferError

import numpy as np

class apfasst_transfer(space_transfer):
    """
    Transfer class for asymptotic PFASST.
    
    Assumes we solve for a single Fourier mode of the linear 1D advection diffusion equation
    
      u_t + U u_x = nu u_xx
      
    Interpolation means multiplication with exp(-U*i*k) whereas restriction means multiplication with exp(U*i*k).
    
    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """
        
        if 'kappa' not in params:
            raise TransferError('Asymptotic PFASST transfer: need parameter kappa corresponding to wave number')
        if 'U' not in params:
            raise TransferError('Asymptotic PFASST transfer: need parameter U corresponding to advection speed')
        
        # invoke super initialization
        super(apfasst_transfer, self).__init__(fine_prob, coarse_prob, params)

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        if isinstance(F, mesh):
            G = mesh(F)
            G.values *= np.exp(-self.params.U*1j*self.params.kappa)
        #elif isinstance(F, rhs_imex_mesh):  ... for now, we only use a fully implicit sweeper
        #    G = rhs_imex_mesh(F)
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        if isinstance(G, mesh):
            F = mesh(G)
            F.values *= np.exp(self.params.U*1j*self.params.kappa)
        #elif isinstance(G, rhs_imex_mesh): ... see above
        #    F = rhs_imex_mesh(G)
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        return F
