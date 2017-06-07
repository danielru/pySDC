import numpy as np
import logging

from pySDC.helpers.pysdc_helper import FrozenClass
import pySDC.helpers.transfer_helper as th
from pySDC.core.Errors import UnlockError
from pySDC.core.BaseTransfer import base_transfer

class apfasst_transfer(base_transfer):
    """
    Standard base_transfer class

    Attributes:
        logger: custom logger for sweeper-related logging
        params(__Pars): parameter object containing the custom parameters passed by the user
        fine (pySDC.Level.level): reference to the fine level
        coarse (pySDC.Level.level): reference to the coarse level
    """

    def __init__(self, fine_level, coarse_level, base_transfer_params, space_transfer_class, space_transfer_params):
        """
        Initialization routine

        Args:
            fine_level (pySDC.Level.level): fine level connected with the base_transfer operations
            coarse_level (pySDC.Level.level): coarse level connected with the base_transfer operations
            base_transfer_params (dict): parameters for the base_transfer operations
            space_transfer_class: class to perform spatial transfer
            space_transfer_params (dict): parameters for the space_transfer operations
        """

        # short helper class to add params as attributes
        class __Pars(FrozenClass):
            def __init__(self, pars):
                self.finter = False
                self.coll_iorder = 1
                self.coll_rorder = 1
                for k, v in pars.items():
                    setattr(self, k, v)

                self._freeze()

        self.params = __Pars(base_transfer_params)

        # set up logger
        self.logger = logging.getLogger('transfer')

        # just copy by object
        self.fine = fine_level
        self.coarse = coarse_level

        # for Q-based transfer, check if we need to add 0 to the list of nodes
        if not self.fine.sweep.coll.left_is_node:
            fine_grid = np.concatenate(([0], self.fine.sweep.coll.nodes))
            coarse_grid = np.concatenate(([0], self.coarse.sweep.coll.nodes))
        else:
            fine_grid = self.fine.sweep.coll.nodes
            coarse_grid = self.coarse.sweep.coll.nodes

        if self.params.coll_iorder > len(coarse_grid):
            self.logger.warning('requested order of Q-interpolation is not valid, resetting to %s' % len(coarse_grid))
            self.params.coll_iorder = len(coarse_grid)
        if self.params.coll_rorder != 1:
            self.logger.warning('requested order of Q-restriction is != 1, can lead to weird behavior!')

        # set up preliminary transfer matrices for Q-based coarsening
        Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=self.params.coll_iorder, pad=0).toarray()
        Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=self.params.coll_rorder, pad=0).toarray()

        # pad transfer matrices if necessary
        if self.fine.sweep.coll.left_is_node:
            self.Pcoll = np.zeros((self.fine.sweep.coll.num_nodes + 1, self.coarse.sweep.coll.num_nodes + 1))
            self.Rcoll = self.Pcoll.T
            self.Pcoll[1:, 1:] = Pcoll
            self.Rcoll[1:, 1:] = Rcoll
        else:
            self.Pcoll = Pcoll
            self.Rcoll = Rcoll

        # set up spatial transfer
        self.space_transfer = space_transfer_class(fine_prob=self.fine.prob, coarse_prob=self.coarse.prob,
                                                   params=space_transfer_params)

    def restrict(self):
        """
        Space-time restriction routine

        The routine applies the spatial restriction operator to teh fine values on the fine nodes, then reevaluates f
        on the coarse level. This is used for the first part of the FAS correction tau via integration. The second part
        is the integral over the fine values, restricted to the coarse level. Finally, possible tau corrections on the
        fine level are restricted as well.
        """
        
        # get data for easier access
        F = self.fine
        G = self.coarse

        # Make sure that coarse and fine time step align
        assert np.isclose(F.status.time, G.status.time, rtol = 1e-10), "Coarse and fine time step do not have the same initial time"
        assert np.isclose(F.dt, G.dt, rtol = 1e-10), "Coarse and fine time step dt are different"
        
        #print "Beginning of time step: %5.3f" % F.status.time
        #print "Length of time step:    %5.3f" % F.dt
        
        fine_nodes_mapped = F.status.time + F.dt*F.sweep.coll.nodes
        #for ttt in fine_nodes_mapped:
        #  print "Fine nodes: %5.3f" % ttt
            
        coarse_nodes_mapped = G.status.time + G.dt*G.sweep.coll.nodes
        #for ttt in coarse_nodes_mapped:
        #  print "Coarse nodes: %5.3f" % ttt

        if np.size(fine_nodes_mapped)==np.size(coarse_nodes_mapped):
          if not np.allclose(fine_nodes_mapped,coarse_nodes_mapped, rtol=1e-10):
            raise NotImplementedError("The APFASST transfer class currently only works if the coarse and fine quadrature nodes are identical")
        else:
          raise NotImplementedError("The APFASST transfer class currently only works if the coarse and fine quadrature nodes are identical")

        #print "\n"
        
        PG = G.prob

        SF = F.sweep
        SG = G.sweep

        # only if the level is unlocked at least by prediction
        if not F.status.unlocked:
            raise UnlockError('fine level is still locked, cannot use data from there')

        # restrict fine values in space
        tmp_u = [self.space_transfer_restrict(F.u[0], F.status.time, F.status.time)]
        for m in range(1, SF.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer_restrict(F.u[m], fine_nodes_mapped[m-1], F.status.time))

        # restrict collocation values
        G.u[0] = tmp_u[0]
        for n in range(1, SG.coll.num_nodes + 1):
            G.u[n] = self.Rcoll[n, 0] * tmp_u[0]
            for m in range(1, SF.coll.num_nodes + 1):
                G.u[n] += self.Rcoll[n, m] * tmp_u[m]

        # re-evaluate f on coarse level
        G.f[0] = PG.eval_f(G.u[0], G.time)
        for m in range(1, SG.coll.num_nodes + 1):
            G.f[m] = PG.eval_f(G.u[m], G.time + G.dt * SG.coll.nodes[m - 1])

        # build coarse level tau correction part
        tauG = G.sweep.integrate()

        # build fine level tau correction part
        tauF = F.sweep.integrate()

        # restrict fine level tau correction part in space
        tmp_tau = []
        for m in range(0, SF.coll.num_nodes):
            tmp_tau.append(self.space_transfer_restrict(tauF[m], fine_nodes_mapped[m], F.status.time))

        # restrict fine level tau correction part in collocation
        tauFG = [tmp_tau[0]]
        for n in range(1, SG.coll.num_nodes):
            tauFG.append(self.Rcoll[n + 1, 1] * tmp_tau[0])
            for m in range(1, SF.coll.num_nodes):
                tauFG[-1] += self.Rcoll[n + 1, m + 1] * tmp_tau[m]

        # build tau correction
        for m in range(SG.coll.num_nodes):
            G.tau[m] = tauFG[m] - tauG[m]

        if F.tau is not None:
            # restrict possible tau correction from fine in space
            tmp_tau = []
            for m in range(0, SF.coll.num_nodes):
                tmp_tau.append(self.space_transfer_restrict(F.tau[m], fine_nodes_mapped[m], F.status.time))

            # restrict possible tau correction from fine in collocation
            for n in range(0, SG.coll.num_nodes):
                G.tau[n] += self.Rcoll[n + 1, 1] * tmp_tau[0]
                for m in range(1, SF.coll.num_nodes):
                    G.tau[n] += self.Rcoll[n + 1, m + 1] * tmp_tau[m]
        else:
            pass


        # save u and rhs evaluations for interpolation
        for m in range(SG.coll.num_nodes + 1):
            G.uold[m] = PG.dtype_u(G.u[m])
            G.fold[m] = PG.dtype_f(G.f[m])

        # works as a predictor
        G.status.unlocked = True

        return None

    def space_transfer_restrict(self, F, t, t0):
      #print "space_transfer_restrict at time: %5.3f" % t
      try:
        U = self.fine.prob.params.U
        kappa = self.fine.prob.params.kappa
      except:
        raise
      F.values *= np.exp(U*1j*kappa*(t-t0))
      return F
    
    def prolong(self):
        """
        Space-time prolongation routine

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        # get data for easier access
        F = self.fine
        G = self.coarse

        PF = F.prob

        SF = F.sweep
        SG = G.sweep
        
        # Make sure that coarse and fine time step align
        assert np.isclose(F.status.time, G.status.time, rtol = 1e-10), "Coarse and fine time step do not have the same initial time"
        assert np.isclose(F.dt, G.dt, rtol = 1e-10), "Coarse and fine time step dt are different"
        
        fine_nodes_mapped = F.status.time + F.dt*F.sweep.coll.nodes
        coarse_nodes_mapped = G.status.time + G.dt*G.sweep.coll.nodes

        if np.size(fine_nodes_mapped)==np.size(coarse_nodes_mapped):
          if not np.allclose(fine_nodes_mapped,coarse_nodes_mapped, rtol=1e-10):
            raise NotImplementedError("The APFASST transfer class currently only works if the coarse and fine quadrature nodes are identical")
        else:
          raise NotImplementedError("The APFASST transfer class currently only works if the coarse and fine quadrature nodes are identical")

        # only of the level is unlocked at least by prediction or restriction
        if not G.status.unlocked:
            raise UnlockError('coarse level is still locked, cannot use data from there')

        # build coarse correction

        # we need to update u0 here for the predictor step, since here the new values for the fine sweep are not
        # received from the previous processor but interpolated from the coarse level.
        # need to restrict F.u[0] again here, since it might have changed in PFASST
        G.uold[0] = self.space_transfer_restrict(F.u[0], F.status.time, F.status.time)

        # interpolate values in space first
        tmp_u = [self.space_transfer_prolong(G.u[0] - G.uold[0], G.status.time, G.status.time)]
        for m in range(1, SG.coll.num_nodes + 1):
            tmp_u.append(self.space_transfer_prolong(G.u[m] - G.uold[m], coarse_nodes_mapped[m-1], G.status.time))

        # interpolate values in collocation
        F.u[0] += tmp_u[0]
        for n in range(1, SF.coll.num_nodes + 1):
            for m in range(0, SG.coll.num_nodes + 1):
                F.u[n] += self.Pcoll[n, m] * tmp_u[m]

        # re-evaluate f on fine level
        F.f[0] = PF.eval_f(F.u[0], F.time)
        for m in range(1, SF.coll.num_nodes + 1):
            F.f[m] = PF.eval_f(F.u[m], F.time + F.dt * SF.coll.nodes[m - 1])

        return None

    def space_transfer_prolong(self, G, t, t0):
      #print "space_transfer_prolong at time: %5.3f" % t
      try:
        U = self.coarse.prob.params.U
        kappa = self.coarse.prob.params.kappa
      except:
        raise
      G.values *= np.exp(-U*1j*kappa*(t-t0))
      return G
    
    def prolong_f(self):
        """
        Space-time prolongation routine w.r.t. the rhs f

        This routine applies the spatial prolongation routine to the difference between the computed and the restricted
        values on the coarse level and then adds this difference to the fine values as coarse correction.
        """

        raise NotImplementedError("For the asymptotic PFASST, f-interpolation is not available")