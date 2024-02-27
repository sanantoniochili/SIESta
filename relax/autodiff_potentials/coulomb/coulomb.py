"""                    COULOMB                    """																			
import torch
from torch import Tensor
import numpy as np
import math
import numpy.typing as npt
from typing import Dict, Tuple

from ..ewaldpotential import *
from ..cutoff import *

class Coulomb(EwaldPotential):

	def __init__(self, chemical_symbols: npt.ArrayLike, charge_dict: Dict, get_shifts: callable):
		super().__init__()
  
		self.alpha = None
		self.real_cut_off = 0
		self.recip_cut_off = 0

		self.chemical_symbols = chemical_symbols.copy()
		self.charges = np.zeros((len(chemical_symbols),))
		self.get_shifts = get_shifts

		count = 0
		for elem in chemical_symbols: 
			# Save charge per ion position according to given charges
			self.charges[count] = charge_dict[elem]
			count += 1

		self.real = None
		self.recip = None
		self.self_energy = None

	
	def set_cutoff_parameters(self, vects: Tensor=None, N: int=0, 
		accuracy: float=1e-21, real: float =0, reciprocal: float=0): 

		if vects is not None:
			volume = torch.det(vects)
			self.alpha = self.get_alpha(N, volume)
			self.real_cut_off = torch.div(math.sqrt(-np.log(accuracy)), self.alpha)
			self.recip_cut_off = torch.mul(self.alpha, 2*math.sqrt(-np.log(accuracy)))
		else:
			self.real_cut_off = real
			self.recip_cut_off = reciprocal


	def ewald_real_energy(self, pos: Tensor, vects: Tensor) -> Tensor:
		shifts = self.get_shifts(vects, self.real_cut_off.item())
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		
		esum = torch.tensor(0.)
		N = len(pos)

		for ioni in range(N):
			for ionj in range(N):
				if ioni != ionj:  # skip in case it's the same ion in original unit cell
					dist = torch.norm(torch.add(pos[ioni], -pos[ionj]))
					term = torch.erfc(torch.mul(self.alpha, dist))
					charges = self.charges[ioni]*self.charges[ionj]
					esum = torch.add(esum, torch.mul(charges, torch.div(term, dist)))

				# Take care of the rest lattice (+ Ln)
				# Start with whole unit cell images
				for shift in range(shifts_no):
					pvec = torch.add(pos[ioni], -pos[ionj])
					dist = torch.norm( torch.add(pvec, shifts[shift]) )
					term = torch.erfc(torch.mul(self.alpha, dist))
					charges = self.charges[ioni]*self.charges[ionj]
					esum = torch.add(esum, torch.mul(charges, torch.div(term, dist)))

		esum = torch.mul(esum, 14.399645351950543/2)  # electrostatic constant
		self.real = esum
		return esum

	
	def ewald_recip_energy(self, pos: Tensor, vects: Tensor, volume: Tensor) -> Tensor:
		if not volume:
			volume = torch.det(vects)

		rvects = self.get_reciprocal_vects(vects)
		shifts = self.get_shifts(rvects, self.recip_cut_off.item())
		if shifts == None:
			shifts_no = 0
		else:
			shifts_no = len(shifts)
		
		esum = torch.tensor(0.)
		N = len(pos)

		for ioni in range(N):
			for ionj in range(N):
       
				# Get distance vector
				rij = torch.add(pos[ioni], -pos[ionj])
    
				for shift in range(shifts_no):
					# shift on 2nd power
					k_2 = torch.dot(shifts[shift], shifts[shift])
					krij = torch.dot(shifts[shift], rij)
					power = -torch.div(k_2, torch.mul(4, torch.pow(self.alpha, 2)))
					cos_term = torch.mul(torch.mul(2*pi, torch.exp(power)), torch.cos(krij))
					# actual calculation
					frac = torch.div(cos_term, torch.mul(k_2, volume))
					charges = self.charges[ioni]*self.charges[ionj]
					esum = torch.add(esum, torch.mul(charges, frac))

		esum = torch.mul(esum, 14.399645351950543)  # electrostatic constant
		self.recip = esum
		return esum


	def ewald_self_energy(self, pos: Tensor) -> Tensor:

		esum = torch.tensor(0.)
		N = len(pos)
  
		for ioni in range(N):
			alphapi = torch.div(self.alpha, math.sqrt(pi))
			esum = torch.add(esum, -torch.mul(self.charges[ioni]**2, alphapi))
   
		esum = torch.mul(esum, 14.399645351950543)  # electrostatic constant
		self.self_energy = esum
		return esum

	def all_energy(self, pos: Tensor, vects: Tensor, volume: Tensor) -> Tensor:
		real_energy = self.ewald_real_energy(pos, vects)	
		recip_energy = self.ewald_recip_energy(pos, vects, volume)
		self_energy = self.ewald_self_energy(pos)
	
		energy = torch.add(real_energy, recip_energy)
		self.energy = torch.add(energy, self_energy)
		return self.energy

