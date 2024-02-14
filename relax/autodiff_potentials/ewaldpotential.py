import numpy as np
import torch, math
from torch import Tensor
from math import pi
from typing import Tuple

class EwaldPotential:
	"""Generic class for defining potentials."""


	def get_reciprocal_vects(self, vects: Tensor) -> Tensor:
		volume = torch.det(vects)
		rvect1 = 2*pi*torch.div(torch.cross(vects[1,:],vects[2,:]), volume)
		rvect2 = 2*pi*torch.div(torch.cross(vects[2,:],vects[0,:]), volume)
		rvect3 = 2*pi*torch.div(torch.cross(vects[0,:],vects[1,:]), volume)
		rvects = torch.cat((rvect1, rvect2, rvect3))
		return torch.reshape(rvects, shape=(3,3))


	def get_alpha(self, N: float, volume: Tensor) -> Tensor:
		accuracy = N**(1/6) * math.sqrt(pi)
		alpha = torch.div(accuracy, torch.pow(volume, 1/3))
		return alpha


	def gradient(energy: Tensor, scaled_pos: Tensor, vects: Tensor, strains: Tensor, volume: Tensor) -> Tuple[Tensor, Tensor]:
		if not volume:
			volume = torch.det(vects)
   
		scaled_grad = torch.autograd.grad(
			energy, 
			(scaled_pos, strains), 
			torch.ones_like(energy), 
			create_graph=True, 
			retain_graph=True,
			materialize_grads=True)
		pos_grad = torch.matmul(scaled_grad[0], torch.inverse(vects))
		strains_grad = torch.div(scaled_grad[1], volume.item())

		return (pos_grad, strains_grad)


	def hessian(grad: Tuple[Tensor, Tensor], scaled_pos: Tensor, vects: Tensor, strains: Tensor, volume: Tensor):
		if not volume:
			volume = torch.det(vects)

		N = len(scaled_pos)
		hessian = torch.tensor(np.zeros((3*N+6, 3*N+6)))
		pos_grad = grad[0]
		for ioni, beta in np.ndindex((len(scaled_pos), 3)):
			partial_pos_i_hessian_scaled  = torch.autograd.grad(
				pos_grad[ioni][beta], 
				(scaled_pos, strains),
				grad_outputs=(torch.ones_like(pos_grad[ioni][beta])),
				create_graph=True, 
				retain_graph=True,
				materialize_grads=True)
			partial_pos_i_hessian = (
				torch.matmul(partial_pos_i_hessian_scaled[0], torch.inverse(vects)),
				torch.div(partial_pos_i_hessian_scaled[1], volume.item())
			)
			for ionj, gamma in np.ndindex(N, 3):
				hessian[3*ioni+beta][3*ionj+gamma] = partial_pos_i_hessian[0][ionj][gamma]
			for straini in range(6):
				hessian[3*ioni+beta][3*N+straini] = partial_pos_i_hessian[1][straini]
   
		strains_grad = grad[1]
		for straini in range(6):
			partial_strain_i_hessian_scaled  = torch.autograd.grad(
				strains_grad[straini], 
				(scaled_pos, strains),
				grad_outputs=(torch.ones_like(strains_grad[straini])),
				create_graph=True, 
				retain_graph=True,
				materialize_grads=True)
			partial_strain_i_hessian = (
				torch.matmul(partial_strain_i_hessian_scaled[0], torch.inverse(vects)),
				partial_strain_i_hessian_scaled[1]
			)
			for ionj, gamma in np.ndindex(N, 3):
				hessian[3*N+straini][3*ionj+gamma] = partial_strain_i_hessian[0][ionj][gamma]
			for strainj in range(6):
				hessian[3*N+straini][3*N+strainj] = partial_strain_i_hessian[1][strainj]
		
		return hessian

	def get_gnorm(grad, N):
		ind = np.tril_indices(3, k=-1)
		unique_grad = grad.copy()
		unique_grad[N+ind[0], ind[1]] = 0
		gnorm = np.sum(unique_grad**2)
		return math.sqrt(gnorm)/(3*N+6)