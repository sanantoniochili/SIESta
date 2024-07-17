import os, pickle
from ase.io import read
from ase.visualize import view

dir = 'output/output_random_1_CG_to_CubicMin/structs/structure_1_O9Sr3Ti3_switch/'
def myprint(I, dir):
    for i in range(I):
        with open(dir+'structure_1_O9Sr3Ti3_switch_'+str(i)+'.pkl', 'rb') as f:
            output = pickle.load(f)
            print('gnorm',output['Gnorm'],
                  'energy',output['Energy'],
                  'step',output['Step'])
                #   'cubic',output['Cubic'], output['Iter'])
    atoms = read(dir+'structure_1_O9Sr3Ti3_switch_'+str(I)+'.cif')
    view(atoms)

if __name__=='__main__':
    iters = int(input('Iterno: '))
    myprint(iters, dir)