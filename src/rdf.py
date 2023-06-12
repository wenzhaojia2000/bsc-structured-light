# -*- coding: utf-8 -*-
"""
Coder: James Jia
Latest Modification: 2022-05-16
Description: Coding of the radial distribution function given by _________
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.formatter.limits"] = [-5,5]

def normalizationFactor(p:int, l:int) -> float:
    '''
    p : Radial Index
    l : Topological charge

    C : Normalization factor
    '''
    Csq = 2**(abs(l) + 1)\
        * np.math.factorial(p)\
        * 1/(np.pi * np.math.factorial(p + abs(l)))
    C = np.sqrt(Csq)
    return C

def radialDistributionFunction(p:int, l:int, w0:float, r:float) -> float:
    '''
    p (int) : Radial Index
    l (int): Topological charge
    w0 : Beam waist at z = 0, in m  
    r : Position, in m

    f : Radial Distribution Function
    '''
    C = normalizationFactor(p, l)
    f = (C/w0)\
      * ((np.sqrt(2) * r)/(w0))**np.abs(l)\
      * np.exp(-r**2/w0**2)\
      * sp.genlaguerre(p, np.abs(l))(2*r**2/w0**2)
    return f

def visualizeRDF(p:int, l:int, w0:float, rmin:float, rmax:float,\
                 saveSVG=False) -> None:
    '''
    function (object): Function to visualize
    p (int): Radial Index
    l (int): Topological charge
    w0 : Beam waist at z = 0, in m
    rmin : Minimum x, y position to plot, in m 
    rmax : Maximum x, y position to plot, in m
    saveSVG (bool): Save an SVG of the result?
    '''
    interval = (rmax - rmin)/200
    x = np.arange(rmin, rmax+interval, interval)
    y = np.arange(rmin, rmax+interval, interval)
    r = x[x>=0]
    x, y = np.meshgrid(x, y)
    rho = np.sqrt(x**2 + y**2)
    line = radialDistributionFunction(p, l, w0, r)
    countour = radialDistributionFunction(p, l, w0, rho)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1plot = ax1.plot(r, line)
    ax1.set_xlim(r[0], r[-1])
    if line[line<=0].size == 0: #set bottom y-lim to 0 if all values are +ive
        ax1.set_ylim(0,)
    ax1.set(xlabel=r'$\rho\,/\,\mathrm{m}$', ylabel=r'$\mathrm{RDF}$')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.ticklabel_format(useMathText=True)
    ax2plot = ax2.contourf(x, y, countour)
    ax2.set_aspect('equal', 'box')
    ax2.set(xlabel=r'$x\,/\,\mathrm{m}$', ylabel=r'$y\,/\,\mathrm{m}$')
    ax2.ticklabel_format(useMathText=True)
    cbar = fig.colorbar(ax2plot, ax=ax2)
    cbar.ax.ticklabel_format(useMathText=True)
    
    fig.set_size_inches(13.5,5)
    fig.set_dpi(200)
    if saveSVG:
        plt.savefig(f"RDF_{p}_{l}.svg", bbox_inches="tight")