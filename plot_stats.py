import os, glob, re, sys, pickle, csv
from os.path import isfile, join
import matplotlib.pyplot as plt

import numpy as np
import argparse
from ase.io import read
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import statistics

import math
from ase.geometry import wrap_positions
import matplotlib.ticker as tick
from matplotlib.ticker import ScalarFormatter
"""
Record information about the structural relaxation
from cif and pickle files.

"""

INTERVAL = 200
MAX_FILES = 1000
MIN_FILES = 0
ITERS = []
PATH = "./"

def iterno(x):
	return(int(x.split('_')[-1].split('.')[0]))


def read_CG_iters_rutile(fig, ax):
	# plot decrease of iters w.r.t to starting
	# iteration number when cubic is used

	df_list = []
	folder_name = "output/output_cubic_fullstep_to_cg_gsb5/structs/"
	folder = os.listdir(folder_name)
	folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	rutile_folders = [x for x in folder if "rutile" in x]

	X,y = [],[]

	for subfolder in rutile_folders:

		struct_folder_name = folder_name+subfolder
		iter_start = int(subfolder.split('_')[-1])

		X.append(iter_start)
		
		# lists of iteration files per structure
		pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
		cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# get no of CG iters
		with open(pkl_list[-1], 'rb') as file:
			output = pickle.load(file)
			y.append(output['Iter'])


	Xnp, ynp = np.array(X), np.array(y)
	ax.plot(Xnp,ynp,label='rutile')


def read_CG_iters_random_gsb10(fig, ax):
	# plot decrease of iters w.r.t to starting
	# iteration number when cubic is used

	folder_name = "output/output_cubic_fullstep_to_cg_gsb10/structs/"
	folder = os.listdir(folder_name)
	folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	srtio_folders = [x for x in folder]

	X,y = [],[]

	for subfolder in srtio_folders:

		struct_folder_name = folder_name+subfolder
		iter_start = int(subfolder.split('_')[-1])

		X.append(iter_start)
		
		# lists of iteration files per structure
		pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
		cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# get no of CG iters
		with open(pkl_list[-1], 'rb') as file:
			output = pickle.load(file)
			y.append(output['Iter'])


	Xnp, ynp = np.array(X), np.array(y)
	ax.plot(Xnp,ynp,label='strontium titanate')


def read_cubic_gnorm_rutile(fig, ax):

	struct_folder_name = "output/output_rutile_1_CubicMin_fullstep/structs/rutile_rattled"

	X,y = [],[]

	# lists of iteration files per structure
	pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
	cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

	# per iteration
	pkl_list = sorted(pkl_list, key=iterno)
	cif_list = sorted(cif_list, key=iterno)

	for pkl_file in pkl_list:
		with open(pkl_file, 'rb') as file:
			output = pickle.load(file)
			X.append(output['Iter'])
			y.append(output['Gnorm'])

	Xnp, ynp = np.array(X), np.array(np.log10(y))
	ax.plot(Xnp,ynp)


def read_cubic_gnorm_random_gsb10(fig, ax):

	struct_folder_name = "output/output_random_1_CubicMin_fullstep/structs/structure_1_O9Sr3Ti3"

	X,y = [],[]

	# lists of iteration files per structure
	pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
	cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

	# per iteration
	pkl_list = sorted(pkl_list, key=iterno)
	cif_list = sorted(cif_list, key=iterno)

	for pkl_file in pkl_list:
		with open(pkl_file, 'rb') as file:
			output = pickle.load(file)
			X.append(output['Iter'])
			y.append(output['Gnorm'])

	Xnp, ynp = np.array(X), np.array(np.log10(y))
	ax.plot(Xnp,ynp)


def plot_rutile_switch_time():
	
	folder_name = "output/output_cubic_fullstep_to_cg_gsb5/structs/"
	folder = os.listdir(folder_name)
	folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	rutile_folders = [x for x in folder if "rutile" in x]

	X,y = [],[]

	for subfolder in rutile_folders:

		time = 0 

		struct_folder_name = folder_name+subfolder
		iter_start = int(subfolder.split('_')[-1])

		if iter_start>400:
			continue

		X.append(iter_start)

		# get time of cubic min iters
		input_file = "output/output_rutile_1_CubicMin_fullstep/structs/rutile_rattled/rutile_rattled_"+str(iter_start)+".pkl"
		with open(input_file, 'rb') as file:
			output = pickle.load(file)
			time += (output['Time'])

		# lists of iteration files per structure
		pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
		cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# get time of CG iters
		with open(pkl_list[-1], 'rb') as file:
			output = pickle.load(file)
			time += (output['Time'])

			y.append(time)

	Xnp, ynp = np.array(X), np.array(y)
	ax.plot(Xnp,ynp,label='rutile')



def plot_random_switch_time():
	
	folder_name = "output/output_cubic_fullstep_to_cg_gsb10/structs/"
	folder = os.listdir(folder_name)
	folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	rutile_folders = [x for x in folder]

	X,y = [],[]

	for subfolder in rutile_folders:

		time = 0 

		struct_folder_name = folder_name+subfolder
		iter_start = int(subfolder.split('_')[-1])

		if iter_start>400:
			continue

		X.append(iter_start)

		# get time of cubic min iters
		input_file = "output/output_random_1_CubicMin_fullstep/structs/structure_1_O9Sr3Ti3/structure_1_O9Sr3Ti3_"+str(iter_start)+".pkl"
		with open(input_file, 'rb') as file:
			output = pickle.load(file)
			time += (output['Time'])

		# lists of iteration files per structure
		pkl_list = glob.glob(struct_folder_name+'/*.{}'.format('pkl'))
		cif_list = glob.glob(struct_folder_name+'/*.{}'.format('cif'))

		# per iteration
		pkl_list = sorted(pkl_list, key=iterno)
		cif_list = sorted(cif_list, key=iterno)

		# get time of CG iters
		with open(pkl_list[-1], 'rb') as file:
			output = pickle.load(file)
			time += (output['Time'])

			y.append(time)

	Xnp, ynp = np.array(X), np.array(y)
	ax.plot(Xnp,ynp,label='strontium titanate')



if __name__=='__main__':

	if not os.path.isdir("plots"):
		os.mkdir("plots")
	
	fig, ax = plt.subplots(figsize=(5,5))
	read_CG_iters_rutile(fig, ax)
	read_CG_iters_random_gsb10(fig, ax)
	ax.set_xlim(0, 400)
	ax.set_title("Steps When Switching \n from CubicRegular to CG", fontsize=15)
	ax.set_xlabel("No. of CubicRegular Steps",  fontsize=13)
	ax.set_ylabel("No. of CG Steps",  fontsize=13)
	ax.legend()
	plt.grid()
	plt.tight_layout()
	fig.savefig("StepsswitchCubictoCG.pdf")
	plt.show()

	# fig, ax = plt.subplots()
	# read_cubic_gnorm_rutile(fig, ax)
	# read_cubic_gnorm_random_gsb10(fig, ax)
	# ax.set_xlim(0, 400)
	# plt.show()

	fig, ax = plt.subplots(figsize=(5,5))
	plot_rutile_switch_time()
	plot_random_switch_time()
	ax.set_xlim(0, 400)
	ax.set_title("Time When Switching \n from CubicRegular to CG", fontsize=15)
	ax.ticklabel_format(style="sci", axis="y", scilimits=(0,2))  
	ax.set_xlabel("Time of CubicRegular Steps [s]",  fontsize=13)
	ax.set_ylabel("Time of CG Steps [s]",  fontsize=13)
	ax.legend()
	plt.grid()
	plt.tight_layout()
	fig.savefig("TimesswitchCubictoCG.pdf")
	plt.show()
