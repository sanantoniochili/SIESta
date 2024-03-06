import math, torch
import numpy as np
from ase.geometry import wrap_positions
from relax.optim.analytic import *
from relax.optim.optimizer import Optimizer
from .cubic_regular.cubicmin import cubic_regularization, cubic_minimization
from .cubic_regular.convexopt import *
from .cubic_regular.prettyprint import PrettyPrint 

pprint = PrettyPrint()

class CubicMin(Optimizer):
    
	def __init__(self, lnsearch, L=1, c=1e+2, inner_tol=1e-3, 
              ftol=1e-5, gtol=1e-3, tol=1e-3, debug=False):
		super().__init__(lnsearch, ftol, gtol, tol)
	
		self.init_L = L
		self.c = c
		self.tol = tol
		self.inner_tol = inner_tol
		self.kappa = math.sqrt(900/(self.tol*L))
		self.reg_value = None
		self.requires_hessian = True
  
		self.debug = True
		if debug:
			pprint.print_emphasis('Running cubic min')
			pprint.print_start({
					'final tol': self.tol,
					'optim tol': inner_tol,
					'c': c,
					'kappa': self.kappa
					})

	
	def completion_check(self, gnorm):
		if super().completion_check(gnorm):
			return True
		if self.reg_value is not None:
			if self.reg_value > - self.tol**(3/2)/(self.c*math.sqrt(self.init_L)):
				if self.debug:
					pprint.print_result({'final': self.reg_value}, 
						tol=- self.tol**(3/2)/(self.c*math.sqrt(self.init_L)),
						iterno=self.iterno)
					pprint.print_emphasis('cubic min end')
				return True
		return False
 

	def step(self, grad, gnorm, hessian, params, line_search_fn, **kwargs):
   
		# Get inner stepsize
		inner_stepsize = 1e-3
		if 'stepize' in kwargs:
			inner_stepsize = kwargs['stepsize']

		grad_vec = np.reshape(grad, newshape=(grad.shape[0]*grad.shape[1],))
		res, L = None, self.init_L

		initial_vector = torch.zeros(hessian.shape[0])
		optimizer = torch.optim.SGD([initial_vector], lr=inner_stepsize)
		optargs = {'params': [initial_vector], 
					'lr': inner_stepsize, 
					'weight_decay': 0,
					'momentum': 0.9,
					'nesterov': True, 
					'maximize': False,
					'foreach': None,
					'dampening': 0.9,
					'differentiable': False}
		res, _ = cubic_minimization(grad=grad_vec, gnorm=gnorm, 
			hessian=hessian, L=L, kappa=self.kappa, 
			optimizer=optimizer, tol=self.inner_tol, 
			debug=self.debug, check=True, **optargs)

		# Calculate cubic regularization function for returned vectors
		reg_value = cubic_regularization(grad_vec, hessian, res[1], L)	
		# Initialize displacement
		displacement = np.reshape(res[1], grad.shape)
		# Check if there is a lowest eigenvector approximation
		if res[2] is not None:
			reg_value_min = res[0]/(2*L)*cubic_regularization(grad_vec, hessian, res[2], L)
			# Keep the vector that gives smaller regular/tion value
			if reg_value_min<reg_value:
				reg_value = reg_value_min
				displacement = np.reshape(res[2], grad.shape)
    
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)
  
		# Add displacement
		params = params + stepsize*displacement
		
		if self.debug:
			pprint.print_result({'current': self.reg_value}, 
						tol=- self.tol**(3/2)/(self.c*math.sqrt(self.init_L)),
						iterno=self.iterno)
			pprint.print_comparison({
				'new cubic regular/tion': reg_value})

		# Save cubic regularization value for termination check
		self.reg_value = reg_value
				
		# Add name of used method to list
		self.iterno += 1

		return params

