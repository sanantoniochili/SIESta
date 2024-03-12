import math, torch
import numpy as np
from ase.geometry import wrap_positions
from relax.optim.analytic import *
from relax.optim.optimizer import Optimizer
from .cubic_regular.cubicmin import cubic_regularization, cubic_minimization
from .cubic_regular.convexopt import *
from .cubic_regular.prettyprint import PrettyPrint 

from .estimator import CubicFit
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingRandomSearchCV
import scipy

pprint = PrettyPrint()

class CubicMin(Optimizer):
    
	def __init__(self, lnsearch, L=1, c=1e+2, inner_tol=1e-3, 
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
			'inner_tol': inner_tol
		}
		self.optargs = {'params': None, 
					'lr': 1e-3, 
					'weight_decay': 0,
					'momentum': 0.5,
					'nesterov': True, 
					'maximize': False,
					'foreach': None,
					'dampening': 0.5,
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
		
		# Try hyperparameter optimization
		if len(self.history) > 9:
			optargs = self.taster_step(kwargs['rng'])
			for key, value in optargs.items():
				if key in self.optargs.keys():
					self.optargs[key] = value
				elif key in self.cparams.keys():
					self.cparams[key] = value
			self.history = []

		# Run fast cubic minimization
		res, _ = cubic_minimization(grad=grad_vec, gnorm=gnorm, 
			hessian=hessian, hnorm=hnorm, 
			L= self.cparams['L'], kappa=self.cparams['kappa'],
			optimizer=optimizer, tol=self.cparams['inner_tol'], 
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


	def taster_step(self, rng):

		X, y = [], []
		for group in self.history:
			grad = group['grad']
			gnorm = group['gnorm']
			gnorm_mat = np.repeat(gnorm, grad.shape)

			hessian = group['hessian']
			hnorm = group['hnorm']
			hnorm_mat = np.repeat(hnorm, grad.shape)

			params = np.asarray([grad, gnorm_mat, hnorm_mat])
			params = np.concatenate((params, hessian))
			X.append(params)
			y.append(0)

		cb = CubicFit(L=10, kappa=30, lr=1, momentum=0, dampening=0, rng=rng)
		param_dist = {
			"L": [1]+[x*10 for x in range(1, 10)], 
			"kappa": rng.randint(low=30, high=90, size=(10,)), 
			"lr": [1e-5, 1e-3, 1e-2, 1e-1, 1], 
			"momentum": [x/10 for x in range(10)], 
			"dampening": [x/10 for x in range(10)],
		}
		rsh = HalvingRandomSearchCV(
			estimator=cb, param_distributions=param_dist, 
			random_state=rng, scoring='max_error',
			aggressive_elimination=True
		)
		rsh.fit(X, y)
		return rsh.best_params_
