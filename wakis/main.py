'''
Wakis main class to manage attributes and methods 
across the modules

It can be instantiated from a previous output file
or constructed through inputs module classes, from 
which it inherits the attributes.

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

import os
import time
import json
import pickle as pk 

#dependencies
import numpy as np 
import matplotlib.pyplot as plt

from wakis.solver import Solver
from wakis.plotting import Plot
from wakis.inputs import Inputs

_cwd = os.getcwd() + '/'

class Wakis(Inputs, Solver, Plot):
    '''
    Central class to manage attributes and methods across the modules

    Attributes include input units, beam parameters and integration  
    path, EM field and charge distribution data, and verbose level
    Methods include setters for atributes and logger initialization
    '''

    def __init__(self, **kwargs):

        #user
        self.unit_m = None  #default: mm
        self.unit_t = None  #default: ns
        self.unit_f = None  #default: GHz
        self.case = None
        self.path = None
        self.verbose = None
        self.log = None

        #beam
        self.q = None
        self.sigmaz = None
        self.xsource, self.ysource = None, None
        self.xtest, self.ytest = None, None
        self.chargedist = None

        #field
        self.Ez = None
        self.t = None
        self.x, self.y, self.z = None, None, None    #field subdomain
        self.x0, self.y0, self.z0 = None, None, None #full simulation domain

        #solver init
        self.s = None
        self.lambdas = None
        self.WP = None
        self.WP_3d = None
        self.WPx, self.WPy = None, None
        self.f = None
        self.Z = None
        self.Zx, self.Zy = None, None
        self.lambdaf = None

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return f'Wakis atributes: \n' + \
        '- beam: \n q={self.q}, sigmaz={self.sigma}, xsource={self.xsource}, ysource={self.ysource}, xtest={self.xtest}, ytest={self.ytest} \n' +\
        '- field: \n Ez={self.Ez}, \n t={self.t}, \n x={self.x}, y={self.y}, z={self.z}, x0={self.x0}, y0={self.y0}, z0={self.z0} \n' + \
        '- charge distribution: \n chargedist={self.chargedist} \n'
        '- solver: \n s={s}, lambdas={self.lambdas}, WP={self.WP}, Z={self.Z}, WPx={self.WPx}, WPy={self.WPy}, Zx={self.Zx}, Zy={self.Zy} \n'

    @classmethod
    def from_inputs(cls, *clss):
        '''
        Factory method from input's module
        classes: User, Beam and Field
        '''
        d = {}
        for cl in clss:
            d.update(cl.__dict__)

        return cls(**d)

    @classmethod
    def from_file(cls, file =  'wakis.json'):
        '''
        Set attributes from output file 'wakis.json'
        '''
        try:
            with open(file, 'r') as f:
                d = {k: np.array(v) for k, v in js.loads(f.read()).items()}

            return cls(**d)

        except:
            self.log.warning(f'"{f}" file not found')

    def solve(self):
        '''
        Perform the wake potential and impedance for
        longitudinal and transverse plane and display
        calculation time
        '''
        t0 = time.time()

        print('---------------------')
        print('|   Running WAKIS   |')
        print('---------------------')

        # Obtain longitudinal Wake potential
        WP_3d, i0, j0 = Solver.calc_long_WP(self)

        #Obtain transverse Wake potential
        Solver.calc_trans_WP(self,WP_3d, i0, j0)

        #Obtain the longitudinal impedance
        Solver.calc_long_Z(self)

        #Obtain transverse impedance
        Solver.calc_trans_Z(self)

        #Elapsed time
        t1 = time.time()
        totalt = t1-t0
        self.log.info('Calculation terminated in %ds' %totalt)

    def save(self, ext = 'json'):
        '''
        Save results in 'wakis' file. 
        Two extensions supported: 'json' and 'pickle'

        parameters
        ----------
        ext : :obj: `str`, optional
            Extention to be used in output file: 'json' or 'pickle'
            Default is 'json'
        '''
        d = self.__dict__

        if ext == 'json':
            j = json.dumps({k: v.tolist() for k, v in d.items()})
            with open('wakis.' + ext, 'w') as f:
                json.dump(j, f)
            self.log.info('"wakis.' + ext +'" file succesfully generated') 

        elif ext == 'pk' or ext == 'pickle':
            with open('wakis.' + ext, 'wb') as f:
                pk.dump(d, f)
            self.log.info('"wakis.' + ext +'" file succesfully generated') 
        
        else: 
            self.log.warning(f'Extension ".{ext}" not supported')


    def plot(self): 
        ''' 
        Plot wakis results in different figures
        that are returned as dictionaries
        '1' : Longitudinal wake potential
        '2' : Longitudinal impedance
        '3' : Transverse wake potentials
        '4' : Transverse impedances
        '5' : Charge distribution
        '''

        figs = {}
        axs = {}

        fig['1'], axs['1'] = Plot.plot_long_WP()
        fig['2'], axs['2'] = Plot.plot_long_Z(plot='all')
        fig['3'], axs['3'] = Plot.plot_trans_WP()
        fig['4'], axs['4'] = Plot.plot_trans_Z(plot='all')
        fig['5'], axs['5'] = Plot.plot_charge_dist()

        return figs, axs

    def subplot(self, save = True):
        ''' 
        Subplot with all wakis results in the same 
        figure and returns each ax as a dictionary:
            '1' : Longitudinal wake potential
            '2' : Longitudinal impedance
            '3' : Transverse wake potentials
            '4' : Transverse impedances
        '''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(16, 10)
        axs = {}

        plt.text(x=0.5, y=0.96, s="WAKIS wake solver result", 
                fontsize='x-large', fontweight='bold', ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.93, s= '(x,y) source = ('+str(round(self.xsource/1e3,1))+','+str(round(self.ysource/1e3,1))+') mm | test = ('+str(round(self.xtest/1e3,1))+','+str(round(self.ytest/1e3,1))+') mm', 
                fontsize='large', ha="center", transform=fig.transFigure)  

        fig, axs['1'] = Plot.plot_long_WP(fig = fig, ax = ax1, chargedist = True)
        fig, axs['2'] = Plot.plot_long_Z(fig = fig, ax = ax1,plot='all')
        fig, axs['3'] = Plot.plot_trans_WP(fig = fig, ax = ax1)
        fig, axs['4'] = Plot.plot_trans_Z(fig = fig, ax = ax1,plot='all')

        if save: fig.savefig(self.path+'wakis.png')

        return fig, axs