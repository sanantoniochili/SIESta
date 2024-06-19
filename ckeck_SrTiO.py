import os, pickle
from ase.io import read
from ase.visualize import view


dir = 'output/output_latlong_CubicMin(not_norm)/structs/SrTiO_latlong/'
def myprint(I, dir):
    for i in range(1,I):
        with open(dir+'SrTiO_latlong_'+str(i)+'.pkl', 'rb') as f:
            output = pickle.load(f)
            print('gnorm',output['Gnorm'],
                  'energy',output['Energy'],
                  'step',output['Step'],
                  'cubic',output['Cubic'])
    atoms = read(dir+'SrTiO_latlong_'+str(I)+'.cif')
    view(atoms)

if __name__=='__main__':
    iters = int(input('Iterno: '))
    myprint(iters, dir)
