import math
import pprint
import numpy as np
from ase.geometry import wrap_positions
from relax.optim.analytic import *
from relax.optim.optimizer import Optimizer
from cubic_regular.cubicmin import cubic_minimization, nesterovAGD, cubic_regularization

class CubicMin(Optimizer):
    
	def __init__(self, lnsearch, L, L2, c, inner_tol, 
              ftol=0.00001, gtol=0.001, tol=0.001, debug=False):
		super().__init__(lnsearch, ftol, gtol, tol)
		# 	def cubic_minimization_wrapper(x, objective, L, L2, kappa, optfunc, 
		# optstep, c, max_iterno=1000, tol_out=None, inner_tol=None, B=None, 
		# debug=False, filename=None, frequency=100) -> np.ndarray:
	
		self.init_L = L
		self.c = c
		self.inner_tol = inner_tol
		self.kappa = math.sqrt(900/(self.tol*L))
		self.reg_value = None
  
		self.debug = debug
		if debug:
			pprint.print_emphasis('Running cubic min')
			pprint.print_start({
					'final tol': self.tol,
					'optim tol': inner_tol,
					'c': c,
					'kappa': self.kappa
					})

	
	def completion_check(self, gnorm):
		if self.reg_value is not None:
			if self.reg_value > - self.tol**(3/2)/(self.c*math.sqrt(self.init_L)):
				if self.debug:
					pprint.print_result({'min final': x}, 
						tol=- self.tol**(3/2)/(self.c*math.sqrt(self.init_L)),
						iterno=self.iterno)
					pprint.print_emphasis('cubic min end')
				return True
		return False
 

	def step(self, grad, gnorm, params, line_search_fn, hessian, L, L2):

		# Minimize cubic regular/tion function
		if self.debug:
			pprint.print_emphasis('iteration no.%d' % self.iterno)
		
		# Calculate stepsize
		lnsearch = getattr(self.lnscheduler, line_search_fn)
		stepsize = lnsearch(gnorm=gnorm, iteration=self.iterno)	

		res, L = None, self.init_L
		while(res is None):
			res, _ = cubic_minimization(grad, hessian, L, L2, self.kappa, 
				nesterovAGD, stepsize, tol=self.inner_tol, debug=False, check=False)
			# If the result is a failed binary search
			# reduce L, otherwise use the L from input
			L = L/2

		# Calculate cubic regularization function for returned vectors
		reg_value = cubic_regularization(grad, hessian, res[1])	
		# Initialize displacement
		displacement = res[1] 
		# Check if there is a lowest eigenvector approximation
		if res[2] is not None:
			reg_value_min = res[0]/(2*L)*cubic_regularization(grad, hessian, res[2])
			# Keep the vector that gives smaller regular/tion value
			if reg_value_min<reg_value:
				reg_value = reg_value_min
				displacement = res[2]

		# Add displacement
		params = params + displacement
		
		if self.debug:
			pprint.print_result({'current min': x}, 
				tol=- self.tol**(3/2)/(self.c*math.sqrt(L)),
				iterno=self.iterno)
			pprint.print_comparison({
				'cubic regular/tion': reg_value})

		# Save cubic regularization value for termination check
		self.reg_value = reg_value
				
		# Add name of used method to list
		self.iterno += 1

		return params

