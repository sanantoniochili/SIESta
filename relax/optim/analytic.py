import argparse, os
import numpy as np
import pickle

from ase.io import read as aread
from ase.cell import Cell
from ase.geometry import wrap_positions, get_distances
from ase.io import write
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from relax.potentials.potential import *
from ..potentials.coulomb.coulomb import Coulomb
from ..potentials.buckingham.buckingham import Buckingham
from ..potentials.cutoff import inflated_cell_truncation

from relax.optim.gradient_descent import GD
from relax.optim.conjugate_gradient import CG
from relax.finite_differences import finite_diff_grad

from relax.linmin import *

import shutil
COLUMNS = shutil.get_terminal_size().columns
def prettyprint(dict_):
	import pprint
	np.set_printoptions(suppress=True)
	words = ""
	for key, value in dict_.items():
		if key=="Total energy":
			words += key+" "+str(value)+" "
		else:
			print("\n", key)
			print(value)
	print(words.center(COLUMNS,"-"))
 
def init(charge_dict, atoms, outdir):

	"""  INITIALISATION  """
	vects 				= np.asarray(atoms.get_cell())
	scaled_pos 			= np.asarray(atoms.get_scaled_positions())
	chemical_symbols	= np.array(atoms.get_chemical_symbols())
	N 					= len(atoms.positions)

	# Define Coulomb potential object
	libfile = "libraries/madelung.lib"
	Cpot = Coulomb(
		chemical_symbols=chemical_symbols,
		N=N,
		charge_dict=charge_dict,
		filename=libfile)
	Cpot.set_cutoff_parameters(
		vects=vects, 
		N=N)
	Cpot.energy(atoms)
	coulomb_energies = Cpot.get_all_ewald_energies()
	
	# Define Buckingham potential object
	libfile = "libraries/buck.lib"
	Bpot = Buckingham(
		filename=libfile, 
		chemical_symbols=chemical_symbols, 
		)
	Bpot.set_cutoff_parameters(
		vects=vects, 
		N=N)
	Bpot.energy(atoms)
	buckingham_energies = Bpot.get_all_ewald_energies()

	# Print Ewald-related parameters
	Cpot.print_parameters()
	Bpot.print_parameters()

	potentials = {}
	initial_energy = 0
	
	potentials['Coulomb'] = Cpot
	potentials['Buckingham'] = Bpot

	if not os.path.isdir(outdir):
		os.mkdir(outdir) 

	prettyprint({
		'Chemical Symbols':chemical_symbols, 
		'Positions':atoms.positions, \
		'Cell':atoms.get_cell(), 
		'Electrostatic energy':coulomb_energies, 
		'Interatomic energy':buckingham_energies, \
		'Total energy':initial_energy})
 
	return potentials, vects, scaled_pos, N
 

# def iter_step(self, atoms, potentials, last_iter={}, 
# 	step_func=steady_step, direction_func=GD, max_step=1, 
# 	update=['ions', 'lattice'], **kwargs):
# 	"""Updating iteration step.

# 	Parameters
# 	----------
# 	atoms : Python ASE's Atoms instance.
# 		Object with the parameters to optimise.
# 	potentials : dict[str, Potential]
# 		Dictionary containing the names and Potential 
# 		instances of the energy functions to be used 
# 		as objective functions.
# 	last_iter : dict[str, _]
# 		Dictionary that holds all information related to 
# 		the previous iteration.
# 	step_func : function
# 		The function to be used for line minimisation.
# 	direction_func : function
# 		The function to be used for the calculation of
# 		the direction vector (optimiser).
# 	max_step : float
# 		The upper bound of the step size.

# 	Returns
# 	-------
# 	dict[str, _]
	
# 	"""

# 	# Get number of ions
# 	N = len(atoms.positions)
	
# 	# Normalise already obtained gradient
# 	if last_iter['Gnorm']>0:
# 		grad_norm = last_iter['Gradient']/last_iter['Gnorm']
# 	else:
# 		grad_norm = 0

# 	# Calculate new energy
# 	res_dict, evals = step_func(
# 		atoms=atoms, 
# 		strains=np.ones((3,3)), 
# 		grad=grad_norm, 
# 		gnorm=last_iter['Gnorm'], 
# 		direction=last_iter['Direction'], 
# 		potentials=potentials, 
# 		init_energy=last_iter['Energy'],
# 		iteration=self.iters, 
# 		max_step=max_step,
# 		min_step=kwargs['min_step'],
# 		step_tol=self.step_tol,
# 		update=update, 
# 		schedule=100,
# 		gmag=self.gmag)

# 	# Calculate new point on energy surface
# 	atoms.set_cell(Cell.new(res_dict['Cell']))
# 	atoms.positions = res_dict['Positions']

# 	# Set cutoff parameters
# 	for name in potentials:
# 		if hasattr(potentials[name], 'set_cutoff_parameters'):
# 			potentials[name].set_cutoff_parameters(res_dict['Cell'],N)	

# 	self.iters += 1
	
# 	# Change method every some iterations
# 	if 'reset' in kwargs:
# 		C = 3*N+9
# 		if kwargs['reset'] & (self.iters % C == 0):
# 			direction_func = GD

# 	# Assign new vectors
# 	pos = np.array(atoms.positions)
# 	vects = np.array(atoms.get_cell())

# 	# Gradient of this point on PES
# 	grad = np.zeros((N+3,3))
# 	for name in potentials:
# 		grad += np.array(potentials[name].gradient(
# 		pos_array=pos, vects_array=vects, N_=N))

# 	# Print numerical derivatives
# 	if 'debug' in kwargs:
# 		if kwargs['debug']:
# 			finite_diff_grad(
# 				atoms, grad, N, res_dict['Strains'], 0.00001, potentials)
	
# 	# Gradient norm
# 	gnorm = get_gnorm(grad,N)
# 	# Gradient norm difference in magnitude with last gnorm
# 	self.gmag = -math.floor(math.log(gnorm, math.e))+math.floor(math.log(last_iter['Gnorm'], math.e))

# 	# Normalise gradient
# 	if gnorm>0:
# 		grad_norm = grad/gnorm
# 	else:
# 		grad_norm = 0

# 	# Named arguments for direction function
# 	args = {
# 		'Residual': -last_iter['Gradient']/last_iter['Gnorm'],
# 		# 'Centered': True,
# 		**last_iter
# 	}

# 	# New direction vector -- the returned dict includes
# 	# all information that need to be passed to the
# 	# next direction calculation
# 	dir_dict = direction_func(grad_norm, **args)
	
# 	iteration = {
# 	'Gradient':grad, **dir_dict, 'Positions':atoms.positions.copy(), 
# 	'Strains':res_dict['Strains'], 'Cell':np.array(atoms.get_cell()), 
# 	'Iter':self.iters, 'Method': self.methods[-1], 
# 	'Step':res_dict['Step'], 'Gnorm':gnorm, 'Energy':res_dict['Energy'], 
# 	'Evaluations':evals, 'Catastrophe': res_dict['Catastrophe']}

# 	# Add name of used method to list
# 	self.methods += [direction_func.__name__]

# 	return iteration

def repeat(atoms, outdir, outfile, charge_dict,
	step_func=steady_step, direction_func=GD, 
	usr_flag=False, max_step=1, out=1, **kwargs):
	"""The function that performs the optimisation. It calls repetitively 
	iter_step for each updating step.

	Parameters
	----------
	init_energy : double
		The energy of the initial configuration.
	atoms : Python ASE's Atoms instance.
		Object with the parameters to optimise.
	potentials : dict[str, Potential]
		Dictionary containing the names and Potential 
		instances of the energy functions to be used 
		as objective functions.
	outdir : str
		Name of the folder to place the output files.
	outfile : str
		Name of the output files.
	step_func : function
		The function to be used for line minimisation.
	direction_func : function
		The function to be used for the calculation of
		the direction vector (optimiser).
	usr_flag : bool
		Flag that is used to stop after each iteration
		and wait for user input, if true.
	max_step : float
		The upper bound of the step size.
	out : int
		Frequency of produced output files -- after 
		how many iterations the ouput should be written
		to a file.

	Returns
	-------
	(int, dict[str, _])
	
	"""
	strains = np.ones((3,3))
	potentials, vects, scaled_pos, N = init(charge_dict, atoms, outdir)
	pos = atoms.get_positions()
	
	initial_energy =0
	for name in potentials:
		if hasattr(potentials[name], 'energy') & \
  			('All' in getattr(potentials[name], 'energy')):
			initial_energy += potentials[name]['All']	

	# count_non = 0
	# total_evals = 1
	# final_iteration = None
	# update = kwargs['params'] if 'params' in kwargs else ['ions', 'lattice']

	# if not os.path.isdir(outdir+"imgs"):
	# 	os.mkdir(outdir+"imgs")
	# if not os.path.isdir(outdir+"structs"):
	# 	os.mkdir(outdir+"structs")
	# if not os.path.isdir(outdir+"imgs/"+outfile+"/"):
	# 	os.mkdir(outdir+"imgs/"+outfile)
	# if not os.path.isdir(outdir+"structs/"+outfile+"/"):
	# 	os.mkdir(outdir+"structs/"+outfile)

	# self.emin = initial_energy
	
	# # Gradient for this point on PES
	# grad = np.zeros((N+3,3))
	# for name in potentials:
	# 	if hasattr(potentials[name], 'gradient'):
	# 		grad += np.array(potentials[name].gradient(
	# 		pos_array=pos, vects_array=vects, N_=N))

	# # Print numerical derivatives
	# if 'debug' in kwargs:
	# 	if kwargs['debug']:
	# 		finite_diff_grad(
	# 			atoms, grad, N, strains, 0.00001, potentials)

	# # Gradient norm
	# gnorm = get_gnorm(grad,N)

	# # Normalise gradient
	# if gnorm>0:
	# 	grad_norm = grad/gnorm
	# else:
	# 	grad_norm = 0
	
	# # Direction -- the returned dict includes
	# # all information that need to be passed to the
	# # next direction calculation
	# if direction_func.__name__=='GD' or direction_func.__name__=='CG':
	# 	dir_dict = GD(grad_norm)
	# 	# Add name of used method to list
	# 	self.methods += ['GD']
	# else:
	# 	dir_dict = direction_func(grad_norm, Positions=pos)
	# 	# Add name of used method to list
	# 	self.methods += [direction_func.__name__]	

	# # Keep info of this iteration
	# iteration = {
	# 'Gradient':grad, **dir_dict, 'Positions':atoms.positions.copy(), 
	# 'Strains':strains, 'Cell':np.array(atoms.get_cell()), 'Iter':self.iters, 
	# 'Step': 0, 'Gnorm':gnorm, 'Energy':init_energy, 'Evaluations':total_evals,
	# 'Catastrophe': 0}

	# final_iteration = iteration

	# # OUTPUT
	# prettyprint(iteration)
	# print("Writing result to file",outfile+"_"+\
	# 	str(self.iters),"...")
	# write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
	# 	str(self.iters)+".png", atoms)
	# write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 	str(self.iters)+".cif", atoms)
	# dict_file = open(
	# 	outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 	str(self.iters)+".pkl", "wb")
	# pickle.dump(
	# 	iteration, 
	# 	dict_file)
	# dict_file.close()

	# if usr_flag:
	# 	usr = input()
	# 	if 'n' in usr:
	# 		return total_evals, iteration

	# # Iterations #
	# i = 1
	# while(True):
	# 	last_iteration = iteration
	# 	iteration = self.iter_step(
	# 		atoms=atoms, 
	# 		potentials=potentials, 
	# 		last_iter=last_iteration, 
	# 		step_func=step_func, 
	# 		direction_func=direction_func, 
	# 		max_step=last_iteration['Step'] if last_iteration['Step'] else max_step,
	# 		update=update,
	# 		**kwargs)

	# 	final_iteration = iteration

	# 	# Keep the newly found energy value
	# 	self.emin = iteration['Energy']
	# 	total_evals += iteration['Evaluations']

	# 	prettyprint(iteration)

	# 	# Check for termination
	# 	if self.completion_check(last_iteration, iteration, N):
	# 		print("Writing result to file",
	# 		outfile+"_"+str(self.iters),"...")
	# 		write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".png", atoms)
	# 		write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".cif", atoms)
	# 		dict_file = open(
	# 			outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".pkl", "wb")
	# 		pickle.dump(
	# 			{**iteration, 'Optimised': True}, 
	# 			dict_file)
	# 		dict_file.close()				
	# 		break
	# 	elif (i%out)==0:
	# 		print("Writing result to file",
	# 		outfile+"_"+str(self.iters),"...")
	# 		write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".png", atoms)
	# 		write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".cif", atoms)
	# 		dict_file = open(
	# 			outdir+"structs/"+outfile+"/"+outfile+"_"+\
	# 			str(self.iters)+".pkl", "wb")
	# 		pickle.dump(
	# 			{**iteration, 'Optimised': False}, 
	# 			dict_file)
	# 		dict_file.close()

	# 	# Count consecutive failed line mins
	# 	if ( last_iteration['Step']==None ) & ( iteration['Step']==None ):
	# 		count_non += 1
	# 		if count_non>4:
	# 			print("Line minimisation failed.")
	# 			break
	# 	else:
	# 		count_non = 0

	# 	# Check if max iteration number is reached
	# 	if i == self.iterno:
	# 		break

	# 	if usr_flag:
	# 		if not 'more' in usr:
	# 			usr = input()
	# 		if 'n' in usr:
	# 			return total_evals, iteration

	# 	i += 1

	# return total_evals, final_iteration
