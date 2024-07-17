import os, pickle
from ase.io import read
from ase.visualize import view


dir = 'output/output_rutile_convexopt_CubicMin_fullstep/structs/rutile_rattled/'
def myprint(I, dir):
    for i in range(1,I):
        with open(dir+'rutile_rattled_'+str(i)+'.pkl', 'rb') as f:
            output = pickle.load(f)
            print('gnorm',output['Gnorm'],
                  'energy',output['Energy'],
                  'step',output['Step'],
                  'cubic',output['Cubic'],
                  'iter',output['Iter'])
    atoms = read(dir+'rutile_rattled_'+str(I)+'.cif')
    view(atoms)

if __name__=='__main__':
    iters = int(input('Iterno: '))
    myprint(iters, dir)
    # finalprint()