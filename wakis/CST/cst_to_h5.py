'''
cst_to_h5.py

File for postprocessing 3d logfiles from cst

--- Reads each 3d logfiles
--- Saves the x, y, z and Ez matrixes for each logfile
--- Saves Ez to a .h5 file where each timestep is a dataset

'''

import numpy as np
import glob, os 
import pickle as pk
import h5py


# To be checked by the user
unit=1e-3
data_path='data/3d_QUAD/'
out_folder='runs/'
n_transverse_cells=6
n_longitudinal_cells=293 #rows/n_transverse_cells**2

# Rename files with E-02, E-03
for file in glob.glob(data_path +'*E-02.txt'): 
    file=file.split(data_path)
    title=file[1].split('_')
    num=title[1].split('E')
    num[0]=float(num[0])/100

    ntitle=title[0]+'_'+str(num[0])+'.txt'
    os.rename(data_path+file[1], data_path+ntitle)

for file in glob.glob(data_path +'*E-03.txt'): 
    file=file.split(data_path)
    title=file[1].split('_')
    num=title[1].split('E')
    num[0]=float(num[0])/1000

    ntitle=title[0]+'_'+str(num[0])+'.txt'
    os.rename(data_path+file[1], data_path+ntitle)


# Create h5 file 
hf_name='cst_Ez_quadrupolar_dh05.h5'

if os.path.exists(out_folder+hf_name):
    os.remove(out_folder+hf_name)

hf_Ez = h5py.File(out_folder+hf_name, 'w')

# Initialize variables
Ez=np.zeros((n_transverse_cells, n_transverse_cells, n_longitudinal_cells))
x=np.zeros((n_transverse_cells))
y=np.zeros((n_transverse_cells))
z=np.zeros((n_longitudinal_cells))

nsteps, i, j, k = 0, 0, 0, 0
skip=-4 #number of rows to skip
rows=skip 
t=[]

# Start scan
for file in sorted(glob.glob(data_path+'*.txt')):
    print('Scanning file '+ file)
    title=file.split(data_path)
    title2=title[1].split('_')
    num=title2[1].split('.txt')
    t.append(float(num[0])*1e-9)

    with open(file) as f:
        for line in f:
            rows+=1
            columns = line.split()

            if rows>=0 and len(columns)>1:
                k=int(rows/n_transverse_cells**2)
                j=int(rows/n_transverse_cells-n_transverse_cells*k)
                i=int(rows-j*n_transverse_cells-k*n_transverse_cells**2) 
                
                Ez[i,j,k]=float(columns[5])
                x[i]=float(columns[0])
                y[j]=float(columns[1])
                z[k]=float(columns[2])

    if nsteps == 0:
        prefix='0'*5
        hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)
    else:
        prefix='0'*(5-int(np.log10(nsteps)))
        hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)

    i, j, k = 0, 0, 0          
    rows=skip
    nsteps+=1

    #close file
    f.close()

#Save x,y,z,t in field data dict
data= { 'x' : x*unit,
        'y' : y*unit, 
        'z' : z*unit,
        't' : np.array(t)
        }

#Save dictionary
with open(out_folder+'field_data.txt', 'wb') as handle:
    pk.dump(data, handle)

hf_Ez.close()    
print('Finished scanning files')
print('Ez field is stored in a matrix with shape '+str(Ez.shape)+' in '+str(int(nsteps))+' datasets')


