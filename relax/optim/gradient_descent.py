import numpy as np
from ase.geometry import wrap_positions
from relax.potentials.potential import *
from relax.linmin import *
from optimizer import Optimizer

class GD(Optimizer):

	def step(self, grad, max_step, min_step, params, 
          line_search_fn=steady_step, **kwargs):

		# Calculate direction
		self.methods += ['GD']
		self.direction = -grad
   
		# Calculate new energy
		stepsize = line_search_fn(
			max_step=max_step,
			min_step=min_step,
			schedule=kwargs['schedule'])		

        # Perform a step (MUST REFORM TO CARTESIAN AND APPLY STRAINS AND CUTOFF)
		params = params + stepsize*self.direction

		# Add name of used method to list
		self.methods += ['GD']
		self.iterno += 1
		self.residual = -grad

		return params

