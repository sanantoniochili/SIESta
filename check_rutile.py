import os, pickle
from ase.io import read
from ase.visualize import view


dir = 'output/output_random_1_CubicMin_bfgs/structs/rutile_rattled/'
def myprint(I, dir):
    for i in range(1,I):
        with open(dir+'rutile_rattled_'+str(i)+'.pkl', 'rb') as f:
            output = pickle.load(f)
            print('gnorm',output['Gnorm'],
                  'energy',output['Energy'],
                  'step',output['Step'],
                  'cubic',output['Cubic'])
    atoms = read(dir+'rutile_rattled_'+str(I)+'.cif')
    view(atoms)

def finalprint():
    with open('output/output_random_1_CubicMin_bfgs/rutile_rattled/rutile_rattled_bfgs_final.pkl', 'rb') as f:
        output = pickle.load(f)
        print(output)
    atoms = read('output/output_random_1_CubicMin_bfgs/rutile_rattled/rutile_rattled_bfgs_final.cif')
    view(atoms)

if __name__=='__main__':
    iters = int(input('Iterno: '))
    myprint(iters, dir)
    finalprint()