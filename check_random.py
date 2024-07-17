import os, pickle
from ase.io import read
from ase.visualize import view

dir = 'output/output_random_convexopt_CubicMin_fullstep/structs/structure_1_O9Sr3Ti3/'
def myprint(I, dir):
    for i in range(1,I):
        with open(dir+'structure_1_O9Sr3Ti3_'+str(i)+'.pkl', 'rb') as f:
            output = pickle.load(f)
            print('gnorm',output['Gnorm'],
                  'energy',output['Energy'],
                  'step',output['Step'],
                  'cubic',output['Cubic'], output['Iter'])
    atoms = read(dir+'structure_1_O9Sr3Ti3_'+str(I)+'.cif')
    view(atoms)

if __name__=='__main__':
    iters = int(input('Iterno: '))
    myprint(iters, dir)