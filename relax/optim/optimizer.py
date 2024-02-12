from relax.optim.linmin import *
from typing import overload

class Optimizer:

    def __init__(self, lnsearch, ftol=0.00001, gtol=0.001, tol=0.001):
        self.ftol = ftol
        self.gtol = gtol
        self.tol = tol
        self.iterno = 0
        self.residual = None
        self.lnscheduler = lnsearch

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
    def step(self, grad, params, line_search_fn, **kwargs):
        ...
   