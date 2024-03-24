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
    
	def __init__(self, lnsearch, L=10, c=1e+2, inner_tol=1e-3, 
              ftol=1e-5, gtol=1e-3, tol=1e-3, debug=False):
		super().__init__(lnsearch, ftol, gtol, tol)
	
		self.reg_value = None
		self.requires_hessian = True
		self.history = []
		self.cparams = {
			'kappa': math.sqrt(900/(self.tol*L)),
			'L': L,
			'c': c,
			'tol': tol,
			'inner_tol': inner_tol,
			'B': None
		}
		self.optargs = {'params': None, 
					'lr': 1e-3, 
					'weight_decay': 0,
					'momentum': 0,
					'nesterov': True, 
					'maximize': False,
					'foreach': None,
					'dampening': 0,
					'differentiable': False,
					'eps': 1e-8,
					'alpha': 0.99,
					'centered': True}
  
		self.debug = True
		if debug:
			pprint.print_emphasis('Running cubic min')
			pprint.print_start({
					'final tol': self.cparams['tol'],
					'optim tol': self.cparams['inner_tol'],
					'c': c,
					'kappa': self.kappa
					})

	
	def completion_check(self, gnorm):
		if super().completion_check(gnorm) & (self.reg_value is not None):
			if self.reg_value > - self.cparams['tol']**(3/2)/(
				self.cparams['c']*math.sqrt(self.init_L)):
				if self.debug:
					pprint.print_result({'final': self.reg_value}, 
						tol=- self.cparams['tol']**(3/2)/(self.cparams['c']*math.sqrt(self.init_L)),
						iterno=self.iterno)
					pprint.print_emphasis('cubic min end')
				return True
		return False
 

	def step(self, grad, gnorm, hessian, hnorm, 
		  params, line_search_fn, **kwargs):

		grad_vec = np.reshape(grad, newshape=(grad.shape[0]*grad.shape[1],))
		res = None

		# Keep history 
		self.history.append({
			'grad': grad_vec,
			'gnorm': gnorm,
			'hessian': hessian,
			'hnorm': hnorm
		})

		# Initialize optimizer arguments
		initial_vector = torch.zeros(hessian.shape[0])
		optimizer = torch.optim.SGD([initial_vector], lr=self.optargs['lr'])
		
		# # Try hyperparameter optimization
		# if len(self.history) > 4:
		# 	optargs = self.taster_step(kwargs['rng'])
		# 	for key, value in optargs.items():
		# 		if key in self.optargs.keys():
		# 			self.optargs[key] = value
		# 		elif key in self.cparams.keys():
		# 			self.cparams[key] = value
		# 	self.history = []

		# Run fast cubic minimization
		res, _ = cubic_minimization(
			grad=grad_vec, gnorm=gnorm, 
			hessian=hessian, hnorm=hnorm, 
			optimizer=optimizer, 
			L= self.cparams['L'], 
			B=self.cparams['B'],
			kappa=self.cparams['kappa'], 
			tol=self.cparams['inner_tol'], 
			debug=self.debug, check=True, rng=kwargs['rng'], 
			**self.optargs)

		# Calculate cubic regularization function for returned vectors
		reg_value = cubic_regularization(
			grad_vec, hessian, res[1], self.cparams['L'])	
		# Initialize displacement
		displacement = np.reshape(res[1], grad.shape)
		# Check if there is a lowest eigenvector approximation
		if res[2] is not None:
			reg_value_min = res[0]/(2*self.cparams['L'])*cubic_regularization(
				grad_vec, hessian, res[2], self.cparams['L'])
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
			pprint.print_result({
					'current regularization': self.reg_value,
					'current gnorm': gnorm}, 
					tol=- self.cparams['tol']**(3/2)/(
						self.cparams['c']*math.sqrt(self.cparams['L'])),
					iterno=self.iterno)
			pprint.print_comparison({
				'new cubic regular/tion': reg_value})

		# Save cubic regularization value for termination check
		self.reg_value = reg_value
				
		# Add name of used method to list
		self.iterno += 1

		return params