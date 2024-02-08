from ase import *

def get_example1():
    name = "O2"
    atoms = Atoms(name, cell=[[4, 0.00, 0.00],	
						[0.00, 4, 0.00],
						[0.00, 0.00, 4]],
                positions=[[1, 1, 1],
                                [3.5, 3.5, 3.5]],
                pbc=True)
    return atoms, name