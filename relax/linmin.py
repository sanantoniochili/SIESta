import argparse, sys
import numpy as np
from math import log, exp

from relax.potentials.potential import *
from relax.potentials.buckingham import *
from relax.potentials.coulomb import *

from relax.potentials.cutoff import check_lattice
from relax.potentials.operations import get_all_distances, get_min_dist

from ase import Atoms
from ase.io import read as aread
from ase.visualize import view as aview
from ase.cell import Cell
from ase.geometry import wrap_positions
from ase.geometry import cell_to_cellpar, cellpar_to_cell

import shutil
from multiprocessing import Pool
from itertools import repeat
COLUMNS = shutil.get_terminal_size().columns


def steady_step(max_step=0.001, **kwargs):
	return max_step


def scheduled_bisection(max_step=1, min_step=1e-5, **kwargs):
	step = max_step
	words = "Using step: "+str(step)

	if (kwargs['iteration']>0) & (kwargs['iteration']%kwargs['schedule']==0):
		step = (max_step+min_step)/2

	return step

def scheduled_exp(max_step=1, min_step=1e-5, **kwargs):	
	step = max_step
	if (kwargs['iteration']>0):
		if step > min_step:
			step = max_step*0.999
	return step	

def gnorm_scheduled_bisection(max_step=1, min_step=1e-5, **kwargs):
	step = max_step
	if kwargs['gmag']>0:
		step = (max_step+min_step)/2

	return step
