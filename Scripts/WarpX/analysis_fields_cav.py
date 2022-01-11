#! /usr/bin/env python

import yt
import os, sys
from scipy.constants import mu_0, pi, c, epsilon_0
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors


# Open the right plot file
for i in np.linspace(0, 1200, 121):
    filename = 'diags/gaussian_beam_square_cav_plt' + str(int(i)).zfill(5)
    ds = yt.load(filename)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    t = ds.current_time.to_value()

    Ex = data['boxlib','Ex'].to_ndarray()
    By = data['boxlib','By'].to_ndarray()
    Ft = Ex - c*By

    vmax=8.5e7
    plt.imshow(Ft[:,32,10:-10], extent=[-54e-3, 54e-3,-32e-3,32e-3])#, vmax=vmax, vmin=-vmax)
    cb = plt.colorbar(label='Ex-cBy    [V/m]')
    plt.xlabel('z    [m]')
    plt.ylabel('x    [m]')

    plt.title('PML J damping')
    plt.savefig('figs_cubic_cav/'+str(int(i)).zfill(5)+'.png')
    cb.remove()
  
