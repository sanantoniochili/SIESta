"""                    BUCKINGHAM                    """																			
import torch
from torch import Tensor
import numpy as np
import math
import numpy.typing as npt
from typing import Dict, Tuple

from ..ewaldpotential import *
from ..cutoff import *

class Buckingham(EwaldPotential):
    
    
    def __init__(self,  filename: str, chemical_symbols: npt.ArrayLike, get_shifts: callable):

        self.alpha = None
        self.real_cut_off = 0
        self.recip_cut_off = 0
        
        self.real = None
        self.recip = None
        self.self_energy = None

        self.chemical_symbols = chemical_symbols.copy()
        self.charges = np.zeros((len(chemical_symbols),))
        self.get_shifts = get_shifts
        self.buck = {}

        if filename:
            try:
                with open(filename, "r") as fin:
                    for line in fin:
                        line = line.split()
                        if (len(line) < 4):
                            continue
                        pair = (min(line[0], line[2]), max(line[0], line[2]))
                        self.buck[pair] = {}
                        self.buck[pair]['par'] = list(map(float, line[4:7]))
                        self.buck[pair]['lo'] = float(line[7])
                        self.buck[pair]['hi'] = float(line[-1])
            except IOError:
                raise FileNotFoundError("No library file found for Buckingham constants.")


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
            
    
    def ewald_self_energy(self, volume: Tensor, N: int) -> Tensor:                           

        esum = torch.tensor(0.)
        
        for ioni in range(N):
            for ionj in range(N):
                pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
                        max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
                if (pair in self.buck):
                    # Pair of ions is listed in parameters file
                    C = self.buck[pair]['par'][2]
                    alphapi = torch.mul(torch.pow(self.alpha, 3), pi**(1.5))
                    Cterm = torch.div(C ,torch.mul(3, volume))
                    esum = torch.sub(esum, torch.mul(Cterm, alphapi))
            pair = (self.chemical_symbols[ioni], self.chemical_symbols[ioni])
            if (pair in self.buck):
                # Pair of ions is listed in parameters file
                C = self.buck[pair]['par'][2] 
                alpha6 = torch.pow(self.alpha, 6)
                esum = torch.add(esum, 
                        torch.div(torch.mul(C, alpha6), 6))
            
        esum = torch.div(esum, 2)
        self.self_energy = esum
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

        for ioni, ionj in np.ndindex((N, N)):
            pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
                    max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
            if (pair in self.buck):
                # Pair of ions is listed in parameters file
                C = self.buck[pair]['par'][2]

                # Get distance vector
                rij = torch.add(pos[ioni], -pos[ionj])

                for shift in range(shifts_no):
                    # shift on 2nd power
                    k_2 = torch.dot(shifts[shift], shifts[shift])
                    k_3 = torch.mul(k_2, torch.sqrt(k_2))
                    krij = torch.dot(shifts[shift], rij)
                    
                    # Cterm is C*pow(pi,1.5)/(12*volume)
                    Cterm = torch.div(C*pi**(1.5), torch.mul(12, volume))
                    # costerm is cos(krij)*k_3
                    costerm = torch.mul(torch.cos(krij), k_3)

                    multiplier = torch.mul(Cterm, costerm)
                    
                    # erfcfrac is sqrt(k_2)/(2*alpha)
                    erfcfrac = torch.div(torch.sqrt(k_2), torch.mul(2, self.alpha))            
                    # erfcterm is sqrt(pi)*erfc(erfcfrac) 
                    sum1 = torch.mul(math.sqrt(pi), torch.erfc(erfcfrac))
                                            
                    # alphadif is (4*pow(alpha,3)/k_3 - 2*alpha/sqrt(k_2))
                    alphadif1 = torch.div(torch.mul(4, torch.pow(self.alpha, 3)), k_3)
                    alphadif2 = torch.div(torch.mul(2, self.alpha), torch.sqrt(k_2))
                    alphadif = torch.subtract(alphadif1, alphadif2)
                    
                    # sum2 is alphadif*expterm
                    frac = -torch.div(k_2, torch.mul(torch.pow(self.alpha, 2), 4))
                    expterm = torch.exp(frac)
                    sum2 = torch.mul(alphadif, expterm)
                                            
                    esum = torch.sub(esum, torch.mul(multiplier, torch.add(sum1, sum2)))

        esum = torch.div(esum, 2)  # electrostatic constant
        self.recip = esum
        return esum


    def ewald_real_energy(self, pos: Tensor, vects: Tensor) -> Tensor:
        shifts = self.get_shifts(vects, self.real_cut_off.item())
        if shifts == None:
            shifts_no = 0
        else:
            shifts_no = len(shifts)
            
        esum = torch.tensor(0.)
        N = len(pos)

        for ioni, ionj in np.ndindex((N, N)):
            # Find the pair we are examining
            pair = (min(self.chemical_symbols[ioni], self.chemical_symbols[ionj]),
                    max(self.chemical_symbols[ioni], self.chemical_symbols[ionj]))
            if (pair in self.buck):
                # Pair of ions is listed in parameters file
                A = self.buck[pair]['par'][0]
                rho = self.buck[pair]['par'][1]
                C = self.buck[pair]['par'][2]

                if ioni != ionj:
                    # Get distance vector
                    dist = torch.norm(torch.add(pos[ioni], -pos[ionj]))

                    # Get repulsion energy
                    frac = -torch.div(dist, rho)
                    esum = torch.add(esum, torch.mul(A, torch.exp(frac)))
                    
                    # Get dispersion with ewald
                    # Cterm is C/pow(dist,6)
                    Cterm = torch.div(C, torch.pow(dist, 6))
                    amuldist = torch.mul(self.alpha, dist)
                    ar2 = torch.pow(amuldist, 2)
                    ar4 = torch.pow(ar2, 2)
                    # alphaterm is (1+pow(alpha*dist,2)+pow(alpha*dist,4)/2)
                    alphaterm = torch.add(torch.add(1, ar2), torch.div(ar4, 2))
                    # expterm is exp(-pow(alpha*dist,2))
                    expterm = torch.exp(-torch.pow(amuldist, 2))
                    
                    term = torch.mul(Cterm, alphaterm)
                    term = torch.mul(term, expterm)
                    esum = torch.sub(esum, term)
                    

                for shift in range(shifts_no):
                    # Get distance vector
                    pvec = torch.add(pos[ioni], -pos[ionj])
                    dist = torch.norm( torch.add(pvec, shifts[shift]) )

                    # Get repulsion energy
                    frac = -torch.div(dist, rho)
                    esum = torch.add(esum, torch.mul(A, torch.exp(frac)))
                    
                    # Get dispersion with ewald
                    # Cterm is C/pow(dist,6)
                    Cterm = torch.div(C, torch.pow(dist, 6))
                    amuldist = torch.mul(self.alpha, dist)
                    ar2 = torch.pow(amuldist, 2)
                    ar4 = torch.pow(ar2, 2)
                    # alphaterm is (1+pow(alpha*dist,2)+pow(alpha*dist,4)/2)
                    alphaterm = torch.add(torch.add(1, ar2), torch.div(ar4, 2))
                    # expterm is exp(-pow(alpha*dist,2))
                    expterm = torch.exp(-torch.pow(amuldist, 2))
                    
                    term = torch.mul(Cterm, alphaterm)
                    term = torch.mul(term, expterm)
                    esum = torch.sub(esum, term)
                        
        esum = torch.div(esum, 2)  # electrostatic constant
        self.real = esum
        return esum
    
    def energy(self, pos: Tensor, vects: Tensor, volume: Tensor) -> Tensor:
        real_energy = self.ewald_real_energy(pos, vects)	
        recip_energy = self.ewald_recip_energy(pos, vects, volume)
        self_energy = self.ewald_self_energy(volume, len(pos))

        energy = torch.add(real_energy, recip_energy)
        energy = torch.add(energy, self_energy)
        return energy