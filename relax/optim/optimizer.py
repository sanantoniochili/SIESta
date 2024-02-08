import numpy as np
from ase.geometry import wrap_positions
from relax.potentials.potential import *
from relax.linmin import *
from typing import overload

class Optimizer:

    def __init__(self, ftol=0.00001, gtol=0.001, tol=0.001):
        self.ftol = ftol
        self.gtol = gtol
        self.tol = tol
        self.iterno = 0
        self.residual = None

    def completion_check(self, gnorm):
        """Function to check if termination conditions have 
        been met.

        Parameters
        ----------
        iteration : dict[str, _]
            Dictionary that holds all information related to 
            the previous iteration.

        Returns
        -------
        bool

        """
        if gnorm<self.gtol:
            print("Iterations: {} Final Gnorm: {} Tolerance: {}".format(
                self.iterno, gnorm, self.gtol))
            return True
        return False

    @overload
    def step(self, grad, max_step, min_step, params, 
            line_search_fn=steady_step, **kwargs):
        ...
        
    # def update_fn(params, stepsize, direction):
    #     # Update ion positions
    #     N = len(params[0])
    #     params[0] = params[0] + stepsize*direction[:N]
    #     # pos_temp = wrap_positions(
    #     #     params[0] + stepsize*direction[:N], kwargs['vects'])

    #     # Update lattice
    #     strains_temp = params[1] + stepsize*direction[N:]
    #     strains = (strains_temp-1)+np.identity(3)
    #     params[0] = pos_temp @ strains.T

    #     # return {'Direction': np.reshape(ndirection, kwargs['Direction'].shape)}
    #     # # Assign parameters calculated with altered volume
    #     # for name in potentials:
    #     # 	if hasattr(potentials[name], 'set_cutoff_parameters'):
    #     # 		potentials[name].set_cutoff_parameters(vects_temp,N)