import math, torch
import numpy as np
from ase.geometry import wrap_positions

from relax.optim.optimizer import Optimizer

from .cubic_regular.cubicmin import cubic_regularization, cubic_minimization
from .cubic_regular.convexopt import *
from .cubic_regular.prettyprint import PrettyPrint 

from typing import Callable
from numpy.typing import ArrayLike

pprint = PrettyPrint()

class CubicMin(Optimizer):
    
	def __init__(self, lnsearch, L=1, c=1, inner_tol=1e-3, 
              ftol=1e-5, gtol=1e-3, tol=1e-3, debug=False):
		super().__init__(lnsearch, ftol, gtol, tol)
	
		self.reg_value = None
		self.requires_hessian = True
		self.requires_lipschitz = True
		self.last_point = None
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
					'nesterov': False, 
					'maximize': False,
					'foreach': None,
					'dampening': 0,
					'differentiable': False,
					# 'eps': 1e-8,
					# 'alpha': 0.99,
					'centered': False}
  
		self.debug = True
		if debug:
			pprint.print_emphasis('Running cubic min')
			pprint.print_start({
					'final tol': self.cparams['tol'],
					'optim tol': self.cparams['inner_tol'],
					'c': c,
					'kappa': self.kappa
					})
			
	
	def completion_check(self, gnorm: float):
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
 

	def step(self, grad: ArrayLike, gnorm: float, hessian: ArrayLike, hnorm: float, 
		  params: ArrayLike, line_search_fn: Callable, **kwargs):

		grad_vec = np.reshape(grad, newshape=(grad.shape[0]*grad.shape[1],))
		res = None

		# Keep history 
		self.history.append({
			'grad': grad_vec,
			'gnorm': gnorm,
			'hessian': hessian,
			'hnorm': hnorm
		})

		# Run fast cubic minimization
		res, _ = cubic_minimization(
			grad=grad_vec, gnorm=gnorm, 
			hessian=hessian, hnorm=hnorm, 
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
		self.last_point = params.copy()
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
	

	def lipschitz_constant_estimation(self, energy_x: float, params_x: ArrayLike, 
								   energy_y: float, params_y: ArrayLike):

		self.cparams['L'] = abs(energy_x-energy_y)/np.linalg.norm(params_x-params_y)
		pass