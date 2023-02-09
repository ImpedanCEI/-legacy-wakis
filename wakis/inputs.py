'''
Classes to manage Wakis inputs

User contains units and solver case selection
and logger initialization
Beam contains beam parameters, integration path 
and charge distribution data.
Field contains domain dimensions, time array 
and pointer to electric field h5 file. 

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

import os
import json as js
import pickle as pk

#dependencies 
import numpy as np
import h5py 

from wakis.logger import get_logger
from wakis.reader import Reader


class Inputs(Reader):
    '''Mixin class to encapsulate all child input classes
    '''

    class User():
        ''' Class to store user input data

        Parameters
        ----------
        case : str
            Choose the EM solver to use as input

        unit_m : str or float
            Dimensional units given as str: 'mm', 'cm', 'dm', or as float: 1e-3 for mm
            Default: 'mm'
        unti_t : str or float
            Time units given as str: 'ns', 'ps', 'ms', 'us', or as float: 1e-9 for ns
            Default: 'ns'
        unit_f: str or float
            Frequency units given as str: 'GHz', 'MHz', 'kHz', or as float: 1e9 for MHz
            Default: 'GHz'
        path : :obj: `str`, optional
            Absolute path to working directory 
            Default is cwd

        Raises
        ------
        AssertionError
            If the case chosen is not in the available solvers list: 'warpx', 'cst'
        TypeError
            If the input is not a 'str' or a 'float'
        '''

        def __init__(self, case = None, unit_m = 1e-3, 
                     unit_t = 1e-9, unit_f = 1e9, path = os.getcwd() + '/', verbose = 2):    
            
            case_list = ['warpx', 'cst']
            assert case is None or case in case_list, \
                AssertionError('Input must be one in: '+ case_list)

            if type(unit_m) is str:
                if unit_m == 'mm': self.unit_m = 1e-3
                if unit_m == 'cm': self.unit_m = 1e-2
                if unit_m == 'dm': self.unit_m = 1e-1

            elif type(unit_m) is float: self.unit_m = unit_m
            else: raise TypeError('Non valid dimensional units. Input type must be "str" or "float"')

            if type(unit_t) is str:
                if unit_t == 'ns': self.unit_t = 1e-9
                if unit_t == 'ps': self.unit_t = 1e-12
                if unit_t == 'ms': self.unit_t = 1e-3
                if unit_t == 'us': self.unit_t = 1e-6

            elif type(unit_t) is float: self.unit_t = unit_t
            else: raise TypeError('Non valid time units. Input type must be "str" or "float"')

            if type(unit_f) is str:
                if unit_f == 'GHz': self.unit_f = 1e9
                if unit_f == 'MHz': self.unit_f = 1e6
                if unit_f == 'kHz': self.unit_f = 1e3

            elif type(unit_f) is float: self.unit_f = unit_f
            else: raise TypeError('Non valid frequency units. Input type must be "str" or "float"')

            self.unit_m = unit_m  #default: mm
            self.unit_t = unit_t  #default: ns
            self.unit_f = unit_f  #default: GHz
            self.case = case
            self.path = path
            self.verbose = _verbose #1: Debug, 2: Info, 3: Warning, 4: Error, 5: Critical):
            self.log = get_logger(level=_verbose)

    class Beam():
        ''' Class to store beam input data

        Parameters
        ----------
        q : float
            Total beam charge in [C]
        sigma_z : float
            Beam sigma in the longitudinal plane, in [m]
        x_source : float
            Beam center x axis [m]
        y_source : float
            Beam center y axis [m]
        x_test : float
            Integration path for wake potential x axis [m]
        y_test : float
            Integration path for wake potential y axis [m]
        chargedist: str or float
            Filename containing the beam charge distribution or ndarray 
        path : :obj: `str`, optional
            Absolute path to working directory 
            Default is cwd
        '''

        def __init__(self, q = None, sigmaz = None, 
                     xsource = None, ysource = None, 
                     xtest = None, ytest = None, 
                     chargedist = None, solver=None, filename='warpx.dat'):

            self.q = None
            self.sigmaz = None
            self.xsource, self.ysource = None, None
            self.xtest, self.ytest = None, None
            self.chargedist = None
            self.verbose = 2 #1: Debug, 2: Info, 3: Warning, 4: Error, 5: Critical):
            self.log = get_logger(level=self.verbose)
            self.filename = filename
            if solver == 'warpx':
                self.initialuze_warpx()

            if solver == 'cst':
                self.initialize_cst()

        def initialize_warpx(self):

                                         
        @classmethod
        def from_WarpX(cls, filename = 'warpx.json'):

            ext = filename.split('.')[-1]

            if ext == 'js' or ext == 'json':
                with open(filename, 'r') as f:
                    d = {k: np.array(v) for k, v in js.loads(f.read()).items()}

                return cls(q = d['q'], sigmaz = d['sigmaz'], 
                     xsource = d['xsource'], ysource = d['ysource'], 
                     xtest = d['xtest'], ytest = d['ytest'],
                     chargedist = d['chargedist'], rho = d['rho'])

            elif ext == 'pk' or ext == 'pickle' or ext == 'inp':
                with open(filename, 'rb') as f:
                    d = pk.load(f)

                return cls(q = d['q'], sigmaz = d['sigmaz'], 
                     xsource = d['xsource'], ysource = d['ysource'], 
                     xtest = d['xtest'], ytest = d['ytest'], 
                     chargedist = d['charge_dist'], rho = d['rho'])

            else:
                get_logger(2).warning('warpx file extension not supported')

        @classmethod
        def from_CST(cls, q = None, sigmaz = None, 
                     xsource = None, ysource = None, 
                     xtest = None, ytest = None, 
                     chargedist = 'lambda.txt', rho = None, path=os.getcwd() + '/'):

            if type(chargedist) is str:
                try:
                    chargedist = Reader.read_cst_1d(chargedist, path = path)
                except:
                    get_logger(2).warning(f'Charge distribution file "{chargedist}" not found')

            return cls(q = q, sigmaz = sigmaz, 
                     xsource = ysource, ysource = ysource, 
                     xtest = xtest, ytest = ytest, 
                     chargedist = chargedist)


        @staticmethod
        def _read_cst_1d(file, path=os.getcwd() + '/'):
            '''
            Read CST plot data saved in ASCII .txt format

            Parameters:
            ---
            file : str
                Name of the .txt file to read. Example: 'lambda.txt' 
            path : :obj: `str`, optional
                Absolute path to file. Deafult is cwd
            '''

            X = []
            Y = []

            i=0
            with open(path+file) as f:
                for line in f:
                    i+=1
                    columns = line.split()

                    if i>1 and len(columns)>1:

                        X.append(float(columns[0]))
                        Y.append(float(columns[1]))

            X=np.array(X)
            Y=np.array(Y)
            return {'X':X , 'Y': Y}


    class Field():

        def __init__(self, Ez = None, t= None, x = None, y = None, z = None, 
                     x0 = None, y0 = None, z0 = None, filename=None):

            self.Ez = Ez
            self.t = t
            self.x, self.y, self.z = x, y, z    #field subdomain
            self.x0, self.y0, self.z0 = x0, y0, z0 #full simulation domain

        @classmethod
        def from_CST(cls, folder = '3d', filename = 'Ez.h5'):
            '''
            Factory method for Field class that pre-processes the
            CST 3D field monitor output and saves it in .h5 file

            Parameters
            ----------
            folder : :obj: `str`, optional
                Name of the folder that contains theoutput .txt files from CST.
                Default '3d'
            
            filename : :obj: `str`, optional
                Name of the output filename containing the Ez field matrix

            Returns
            -------
            Field : obj
                Class containing the field inputs

            Outputs
            --------
            Ez.h5: :obj:`.h5`
                HDF5 file containing the Ez(x,y,z) matrix for every timestep
            '''

            path = os.getcwd() + '/'
            path_3d = os.getcwd() + '/' + folder + '/'

            #read CST field monitor output and turn it into .h5 file
            Reader.read_cst_3d(path, path_3d, filename)

            #get field content from h5 file
            hf, dataset = Reader.read_Ez(path, filename)

            #return class
            return cls(Ez = {'hf' : hf, 'dataset' : dataset}, t=t, x=x, y=y, z=z)

        @classmethod
        def from_WarpX(cls, path = os.getcwd() + '/', warpx_filename = 'warpx.json', Ez_filename = 'Ez.h5'):
            
            hf, dataset = Reader.read_Ez(path, Ez_filename)
            ext = warpx_filename.split('.')[-1]
            supported_extensions = {'pk', 'pickle', 'dat', 'out', 'inp'}
            if ext == 'json':
                with open(warpx_filename, 'r') as f:
                    d = {k: np.array(v) for k, v in js.loads(f.read()).items()}

                return cls(Ez = {'hf' : hf, 'dataset' : dataset}, t=d['t'], 
                            x=d['x'], y=d['y'], z=d['z'], 
                            x0=d['x0'], y0=d['y0'], z0=['z0'])

            elif ext in supported_extensions:
                with open(warpx_filename, 'rb') as f:
                    d = pk.load(f)

                return cls(Ez = {'hf' : hf, 'dataset' : dataset}, t=d['t'], 
                            x=d['x'], y=d['y'], z=d['z'], 
                            x0=d['x0'], y0=d['y0'], z0=['z0'])

            else:
                self.log.warning('warpx file extension not supported')


        @staticmethod
        def _read_Ez(path = os.getcwd() + '/', filename = 'Ez.h5'):
            '''
            Read the Ez.h5 file containing the Ez field information
            '''

            hf = h5py.File(path+filename, 'r')
            self.log.info('Reading the h5 file: ' + path + filename + ' ...')
            self.log.debug('Size of the file: ' + str(round((os.path.getsize(path+filename)/10**9),2))+' Gb')

            # get number of datasets
            size_hf=0.0
            dataset=[]
            n_step=[]
            for key in hf.keys():
                size_hf+=1
                dataset.append(key)
                n_step.append(int(key.split('_')[1]))

            return hf, dataset

        @staticmethod
        def _read_cst_3d(path = os.getcwd() + '/', path_3d = '3d', filename = 'Ez.h5'):
                    # Rename files with E-02, E-03
            for file in glob.glob(path_3d +'*E-02.txt'): 
                file=file.split(path_3d)
                title=file[1].split('_')
                num=title[1].split('E')
                num[0]=float(num[0])/100

                ntitle=title[0]+'_'+str(num[0])+'.txt'
                os.rename(path_3d+file[1], path_3d+ntitle)

            for file in glob.glob(path_3d +'*E-03.txt'): 
                file=file.split(path_3d)
                title=file[1].split('_')
                num=title[1].split('E')
                num[0]=float(num[0])/1000

                ntitle=title[0]+'_'+str(num[0])+'.txt'
                os.rename(path_3d+file[1], path_3d+ntitle)

            for file in glob.glob(path_3d +'*_0.txt'): 
                file=file.split(path_3d)
                title=file[1].split('_')
                num=title[1].split('.')
                num[0]=float(num[0])

                ntitle=title[0]+'_'+str(num[0])+'.txt'
                os.rename(path_3d+file[1], path_3d+ntitle)

            fnames = sorted(glob.glob(path_3d+'*.txt'))

            #Get the number of longitudinal and transverse cells used for Ez
            i=0
            with open(fnames[0]) as f:
                lines=f.readlines()
                n_rows = len(lines)-3 #n of rows minus the header
                x1=lines[3].split()[0]

                while True:
                    i+=1
                    x2=lines[i+3].split()[0]
                    if x1==x2:
                        break

            n_transverse_cells=i
            n_longitudinal_cells=int(n_rows/(n_transverse_cells**2))

            # Create h5 file 
            if os.path.exists(path+filename):
                os.remove(path+filename)

            hf = h5py.File(path+filename, 'w')

            # Initialize variables
            Ez=np.zeros((n_transverse_cells, n_transverse_cells, n_longitudinal_cells))
            x=np.zeros((n_transverse_cells))
            y=np.zeros((n_transverse_cells))
            z=np.zeros((n_longitudinal_cells))
            t=[]

            nsteps, i, j, k = 0, 0, 0, 0
            skip=-4 #number of rows to skip
            rows=skip 

            # Start scan
            for file in fnames:
                self.log.debug('Scanning file '+ file + '...')
                title=file.split(path)
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
                    hf.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)
                else:
                    prefix='0'*(5-int(np.log10(nsteps)))
                    hf.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)

                i, j, k = 0, 0, 0          
                rows=skip
                nsteps+=1

                #close file
                f.close()

            hf.close()

            #set field info
            self.log.debug('Ez field is stored in a matrix with shape '+str(Ez.shape)+' in '+str(int(nsteps))+' datasets')
            self.log.info('Finished scanning files - hdf5 file'+filename+'succesfully generated')