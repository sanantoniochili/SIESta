import numpy as np
from ase.geometry import wrap_positions
from relax.optim.analytic import *
from relax.optim.linmin import LnSearch
from relax.optim.optimizer import Optimizer

class GD(Optimizer):

	def step(self, grad, gnorm, params, line_search_fn):

		# Calculate direction
		self.direction = -grad
   
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)	

        # Perform a step
		params = params + stepsize*self.direction

		# Add name of used method to list
		self.iterno += 1

		return params

