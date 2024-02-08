import numpy as np
from ase.geometry import wrap_positions
from relax.potentials.potential import *
from relax.linmin import *
from optimizer import Optimizer

class CG(Optimizer):

	def step(self, grad, max_step, min_step, params, 
          update_fn, line_search_fn=steady_step, **kwargs):

		# Calculate direction
		if self.iterno == 0:
			self.methods += ['GD']
			self.direction = -grad
		else:
			residual = -np.reshape(grad,
				(grad.shape[0]*grad.shape[1],1))
			last_residual = np.reshape(self.residual, 
				(self.residual.shape[0]*self.residual.shape[1],1))
			last_direction = np.reshape(self.direction, 
				(self.direction.shape[0]*self.direction.shape[1],1))
			beta = residual.T @ (residual-last_residual) / (last_residual.T @ last_residual)
			self.direction = residual + beta*last_direction
   
		# Calculate new energy
		stepsize = line_search_fn(
			max_step=max_step,
			min_step=min_step,
			schedule=kwargs['schedule'])		
	
		# Perform a step (MUST REFORM TO CARTESIAN AND APPLY STRAINS AND CUTOFF)
		params = params + stepsize*self.direction

		# Add name of used method to list
		self.methods += ['CG']
		self.iterno += 1
		self.residual = -grad

		return params

