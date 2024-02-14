import os, torch
import numpy as np
import pickle

from ase.io import read as aread
from ase.cell import Cell
from ase.geometry import wrap_positions, get_distances
from ase.io import write
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from relax.autodiff_potentials.coulomb.coulomb import Coulomb
from relax.autodiff_potentials.buckingham.buckingham import Buckingham
from relax.autodiff_potentials.cutoff import inflated_cell_truncation
from relax.autodiff_potentials.ewaldpotential import EwaldPotential
from relax.finite_differences import finite_diff_grad

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
    N 					= len(atoms.positions)
    strains_vec			= torch.tensor([1.,0.,0.,1.,0.,1.], dtype=torch.float64, requires_grad=True)
    strains				= torch.eye(3, dtype=torch.float64)
    vects_np 			= np.asarray(atoms.get_cell())
    vects 				= torch.tensor(vects_np, dtype=torch.float64, requires_grad=False)
    scaled_pos_np 		= np.asarray(atoms.get_scaled_positions())
    scaled_pos 			= torch.tensor(scaled_pos_np, requires_grad=True)
    accuracy			= 0.000000000000000000001
    chemical_symbols	= np.array(atoms.get_chemical_symbols())

    # Apply strains
    ind = torch.triu_indices(row=3, col=3, offset=0)
    strains[ind[0], ind[1]] = strains_vec
    vects				= torch.matmul(vects, torch.transpose(strains, 0, 1))
    pos 				= torch.matmul(scaled_pos, vects)
    volume 				= torch.det(vects)
    
    # Avoid truncating too many terms
    assert((-np.log(accuracy)/N**(1/6)) >= 1)	

    """  DEFINITIONS  """  	
    # Define Coulomb potential object
    Cpot = Coulomb(
        chemical_symbols=chemical_symbols,
        charge_dict=charge_dict,
        get_shifts=inflated_cell_truncation
    )
    Cpot.set_cutoff_parameters(
        vects=vects, 
        N=N)
    coulomb_energy = Cpot.energy(pos, vects, volume)

    # Define Buckingham potential object
    Bpot = Buckingham(
        filename='libraries/buck.lib',
        chemical_symbols=chemical_symbols,
        get_shifts=inflated_cell_truncation
    )
    Bpot.set_cutoff_parameters(
        vects=vects, 
        N=N)
    buckingham_energy = Bpot.energy(pos, vects, volume)

    potentials = {}	
    potentials['Coulomb'] = Cpot
    potentials['Buckingham'] = Bpot
    initial_energy = torch.add(coulomb_energy, buckingham_energy)

    if not os.path.isdir(outdir):
        os.mkdir(outdir) 

    prettyprint({
        'Chemical Symbols':chemical_symbols, 
        'Positions':atoms.positions, \
        'Cell':atoms.get_cell(), 
        'Electrostatic energy':coulomb_energy, 
        'Interatomic energy':buckingham_energy, \
        'Total energy':initial_energy})

    return potentials, strains_vec, strains, \
        vects, scaled_pos, initial_energy


def repeat(atoms, outdir, outfile, charge_dict, line_search_fn,
    optimizer, usr_flag=False, out=1, **kwargs):
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
    res                 = init(charge_dict, atoms, outdir)
    potentials          = res[0]
    strains_vec         = res[1]
    strains             = res[2]
    vects               = res[3]
    scaled_pos          = res[4]
    energy              = res[5]
    pos 				= torch.matmul(scaled_pos, vects)
    volume 				= torch.det(vects)
    N = len(pos)

    final_iteration = None
    history = []

    if not os.path.isdir(outdir+"imgs"):
        os.mkdir(outdir+"imgs")
    if not os.path.isdir(outdir+"structs"):
        os.mkdir(outdir+"structs")
    if not os.path.isdir(outdir+"imgs/"+outfile+"/"):
        os.mkdir(outdir+"imgs/"+outfile)
    if not os.path.isdir(outdir+"structs/"+outfile+"/"):
        os.mkdir(outdir+"structs/"+outfile)

    # Gradient for current point on PES
    grad_tensor = EwaldPotential.gradient(
            energy, scaled_pos, vects, strains_vec, volume)
    # Gradient to numpy and free tensor
    pos_grad = grad_tensor[0].detach().numpy().copy()
    strains_grad = grad_tensor[1].detach().numpy().copy()
    # All to one array
    grad = np.zeros((N+3,3))
    grad[:N, :] = pos_grad
    ind = np.triu_indices(3)
    grad[N+ind[0], ind[1]] = strains_grad
    grad[N+1][0] = grad[N][1]
    grad[N+2][1] = grad[N+1][2]
    grad[N+2][0] = grad[N][2]
    
    # Gradient norm
    gnorm = EwaldPotential.get_gnorm(grad, N)
    optimizer.lnscheduler.gnorm = gnorm

    # Normalise gradient
    if gnorm>0:
        grad_norm = grad/gnorm
    else:
        grad_norm = 0

    # Keep info of this iteration
    iteration = {
    'Gradient':grad, 'Positions':pos, 'Strains':np.ones((3,3)), 
    'Cell':np.array(atoms.get_cell()), 'Iter':optimizer.iterno, 
    'Step': 0, 'Gnorm':gnorm, 'Energy':energy
    }

    # Iterations
    while(True):
        final_iteration = iteration

        # Check for termination
        prettyprint(iteration)
        if optimizer.completion_check(gnorm):
            print("Writing result to file",
            outfile+"_"+str(optimizer.iterno),"...")
            write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".png", atoms)
            write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".cif", atoms)
            dict_file = open(
                outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".pkl", "wb")
            pickle.dump(
                {**iteration, 'Optimised': True}, 
                dict_file)
            dict_file.close()				
            break
        elif (optimizer.iterno%out)==0:
            print("Writing result to file",
            outfile+"_"+str(optimizer.iterno),"...")
            write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".png", atoms)
            write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".cif", atoms)
            dict_file = open(
                outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".pkl", "wb")
            pickle.dump(
                {**iteration, 'Optimised': False}, 
                dict_file)
            dict_file.close()
        elif (('iterno' in kwargs) & (kwargs['iterno'] >= optimizer.iterno)):
            break

        if usr_flag:
            usr = input()
            if 'n' in usr:
                return iteration
            
        # Tensors to numpy
        pos_np = pos.detach().numpy().copy()
        vects_np = vects.detach().numpy().copy()
        
        # Delete the tensors
        del energy
        del pos
        del vects
        del volume
        del strains
        del strains_vec

        ''' 1 --- Apply an optimization step --- 1 '''
        params = np.concatenate((pos_np, np.ones((3,3))), axis=0)
        params = optimizer.step(grad_norm, gnorm, params, line_search_fn)

        # Make a method history
        history.append(type(optimizer).__name__)

        ''' 2 --- Update parameters --- 2 '''
        # Make sure ions stay in unit cell
        pos_temp = wrap_positions(params[:N], vects_np)
        # Update strains
        strains_np = (params[N:]-1)+np.identity(3)

        # Apply strains to all unit cell vectors
        pos_np = pos_temp @ strains_np.T
        vects_np = vects_np @ strains_np.T

        # Calculate new point on energy surface
        atoms.positions = pos_np
        atoms.set_cell(vects_np)
        
        # Get new tensors
        strains_vec = torch.tensor(strains_np[np.triu_indices(3)], requires_grad=True)
        strains		= .5*torch.ones((3,3))+ .5*torch.eye(3, dtype=torch.float64)
        ind = torch.triu_indices(row=3, col=3, offset=0)
        strains[ind[0], ind[1]] = strains[ind[0], ind[1]]*strains_vec
        strains[1][0]       = strains[0][1]
        strains[2][1]       = strains[1][2]
        strains[2][0]       = strains[0][2]
        
        vects 				= torch.tensor(vects_np, dtype=torch.float64, requires_grad=False)
        scaled_pos 			= torch.tensor(pos_np @ np.linalg.inv(vects), requires_grad=True)
        vects				= torch.matmul(vects, torch.transpose(strains, 0, 1))
        pos 				= torch.matmul(scaled_pos, vects)
        volume 				= torch.det(vects)

        # Assign parameters calculated with altered volume
        for name in potentials:
            if hasattr(potentials[name], 'set_cutoff_parameters'):
                potentials[name].set_cutoff_parameters(vects, N)

        ''' 3 --- Re-calculate energy --- 3 '''
        # Calculate energy on current PES point
        energy = torch.tensor(0.)
        for name in potentials:
            if hasattr(potentials[name], 'energy'):
                energy = torch.add(energy, 
                            potentials[name].energy(pos, vects, volume))

        ''' 4 --- Re-calculate derivatives --- 4 '''
        # Gradient for current point on PES
        grad_tensor = EwaldPotential.gradient(
                energy, scaled_pos, vects, strains_vec, volume)
        # Gradient to numpy and free tensor
        pos_grad = grad_tensor[0].detach().numpy().copy()
        strains_grad = grad_tensor[1].detach().numpy().copy()
        # All to one array
        grad = np.zeros((N+3,3))
        grad[:N, :] = pos_grad
        ind = np.triu_indices(3)
        grad[N+ind[0], ind[1]] = strains_grad
        grad[N+1][0] = grad[N][1]
        grad[N+2][1] = grad[N+1][2]
        grad[N+2][0] = grad[N][2]
        
        if abs(grad[0][0]) < 1e-7:
            pass
        
        # Gradient norm
        gnorm = EwaldPotential.get_gnorm(grad, N)
        optimizer.lnscheduler.gnorm = gnorm

        # Normalise gradient
        if gnorm>0:
            grad_norm = grad/gnorm
        else:
            grad_norm = 0
            
        iteration = {
        'Gradient':grad, 'Positions':atoms.positions.copy(), 
        'Strains':params[N:], 'Cell':np.array(atoms.get_cell()), 
        'Iter':optimizer.iterno, 'Method': history[-1], 
        'Step':optimizer.lnscheduler.curr_step, 'Gnorm':gnorm, 'Energy':energy}

    return final_iteration
