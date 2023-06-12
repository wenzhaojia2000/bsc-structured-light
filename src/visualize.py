# -*- coding: utf-8 -*-
"""
Coder: James Jia
Description: Visualization of functions in paraxial.py and nonparaxial.py using
             matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10

def visualizeFunct(funct:object, p:int, l:int, w0:float, k:float, sigma:int,
    rmin:float, rmax:float, mu:"np.array"=np.array([1, 0, 0]),
    Q:"np.array"=np.array([[1,0.001,0.001], [0.001,1,0.001], [0.001,0.001,1]]),
    zticklabel:str="", rfzseperate:bool=False, e1e2seperate:bool=False,
    saveSVG:bool=False) -> None:
    '''
    p (int): Radial Index  
    l (int): Topological charge  
    w0 : Beam waist at z = 0, in m
    k : Wavenumber
    sigma : polarization
    rmin : Minimum x, y position to plot, in m 
    rmax : Maximum x, y position to plot, in m
    mu (opt., vect.): electric dipole transition moment
    Q (opt., matr.): quadrupole transition moment
    zticklabel (str.): the label for the colourbar
    rfzseperate (bool): display [rho,phi,z] contributions as seperate graphs
        (paraxial only)
    e1e2seperate (bool): display bar(e)1e2, e1bar(e)2 as seperate graphs
        (paraxial only)
    saveSVG (bool): Save an SVG of the result?
    '''
    interval = (rmax - rmin)/200
    x = np.arange(rmin, rmax+interval, interval)
    y = np.arange(rmin, rmax+interval, interval)
    x, y = np.meshgrid(x, y)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    res = funct(p, l, w0, k, sigma, rho, phi, mu=mu, Q=Q,\
                rfzseperate=rfzseperate, e1e2seperate=e1e2seperate)
    
    if res.ndim == 2:
        axplot = plt.contourf(x, y, res,
                              levels=np.linspace(np.min(res),np.max(res),15)) 
        ax = plt.gca()
        fig = plt.gcf()
        ax.set_aspect('equal', 'box')
        ax.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$')
        ax.ticklabel_format(useMathText=True)
        cbar = plt.colorbar(axplot)
        cbar.set_label(zticklabel)
        cbar.ax.ticklabel_format(useMathText=True)
        fig.set_size_inches(6,4.5)
    elif rfzseperate:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        try:
            ax1plot = ax1.contourf(x, y, res[0],
                          levels=np.linspace(np.min(res[0]),np.max(res[0]),15))
        except:
            ax1plot = ax1.contourf(x, y, res[0])
        ax1.set_aspect('equal', 'box')
        ax1.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'$\rho$ contribution')
        ax1.ticklabel_format(useMathText=True)
        cbar1 = fig.colorbar(ax1plot, ax=ax1)
        cbar1.set_label(zticklabel)
        cbar1.ax.ticklabel_format(useMathText=True)
        try:
            ax2plot = ax2.contourf(x, y, res[1],
                          levels=np.linspace(np.min(res[1]),np.max(res[1]),15))
        except:
            ax2plot = ax2.contourf(x, y, res[1])
        ax2.set_aspect('equal', 'box')
        ax2.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'$\varphi$ contribution')
        ax2.ticklabel_format(useMathText=True)
        cbar2 = fig.colorbar(ax2plot, ax=ax2)
        cbar2.set_label(zticklabel)
        cbar2.ax.ticklabel_format(useMathText=True)
        try:
            ax3plot = ax3.contourf(x, y, res[2],
                          levels=np.linspace(np.min(res[2]),np.max(res[2]),15))
        except:
            ax3plot = ax3.contourf(x, y, res[2])
        ax3.set_aspect('equal', 'box')
        ax3.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'$z$ contribution')
        ax3.ticklabel_format(useMathText=True)
        cbar3 = fig.colorbar(ax3plot, ax=ax3)
        cbar3.set_label(zticklabel)
        cbar3.ax.ticklabel_format(useMathText=True)
        fig.set_size_inches(22.5,5.5)
    elif e1e2seperate:
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
        ax1plot = ax1.contourf(x, y, np.real(res[0]),
                               levels=np.linspace(np.min(np.real(res[0])),
                                                  np.max(np.real(res[0])),
                                                  15))
        ax1.set_aspect('equal', 'box')
        ax1.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'Re($\bar{\mathrm{E}}1\mathrm{E}2$) contribution')
        ax1.ticklabel_format(useMathText=True)
        cbar1 = fig.colorbar(ax1plot, ax=ax1)
        cbar1.set_label(zticklabel)
        cbar1.ax.ticklabel_format(useMathText=True)
        ax2plot = ax2.contourf(x, y, np.imag(res[0]),
                               levels=np.linspace(np.min(np.imag(res[0])),
                                                  np.max(np.imag(res[0])),
                                                  15))
        ax2.set_aspect('equal', 'box')
        ax2.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'Im($\bar{\mathrm{E}}1\mathrm{E}2$) contribution')
        ax2.ticklabel_format(useMathText=True)
        cbar2 = fig.colorbar(ax2plot, ax=ax2)
        cbar2.set_label(zticklabel)
        cbar2.ax.ticklabel_format(useMathText=True)
        ax3plot = ax3.contourf(x, y, np.real(res[1]),
                               levels=np.linspace(np.min(np.real(res[1])),
                                                  np.max(np.real(res[1])),
                                                  15))
        ax3.set_aspect('equal', 'box')
        ax3.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'Re($\mathrm{E}1\bar{\mathrm{E}}2$) contribution')
        ax3.ticklabel_format(useMathText=True)
        cbar3 = fig.colorbar(ax3plot, ax=ax3)
        cbar3.set_label(zticklabel)
        cbar3.ax.ticklabel_format(useMathText=True)
        ax4plot = ax4.contourf(x, y, np.imag(res[1]),
                               levels=np.linspace(np.min(np.imag(res[1])),
                                                  np.max(np.imag(res[1])),
                                                  15))
        ax4.set_aspect('equal', 'box')
        ax4.set(xlabel=r'$x\,/\,\mathrm{m}$',ylabel=r'$y\,/\,\mathrm{m}$',
                title=r'Im($\mathrm{E}1\bar{\mathrm{E}}2$) contribution')
        ax4.ticklabel_format(useMathText=True)
        cbar4 = fig.colorbar(ax4plot, ax=ax4)
        cbar4.set_label(zticklabel)
        cbar4.ax.ticklabel_format(useMathText=True)
        fig.set_size_inches(18,15)
    # Image parameters
    fig.set_dpi(200)
    if saveSVG:
        plt.savefig(f"{funct.__module__}.{funct.__name__}_{p}_{l}_{sigma}.svg",
                    bbox_inches="tight")

if __name__ == "__main__":
    import rdf
    import paraxial as parax
    import nonparaxial as nonparax

    p = 0
    l = 1
    sigma = 1
    k = 2*np.pi/729e-9
    w0 = 729e-9
    lim = 2e-6
    mu = np.array([1, 0, 0])
    Q = 1e-3*np.array([[1,1e-3,1e-3], [1e-3,-0.5,1e-3], [1e-3,1e-3,-0.5]])
    
    # rdf.visualizeRDF(p, l, w0, -lim, lim, False)
    visualizeFunct(nonparax.e1e2Term, p, l, w0, k, sigma, -lim, lim, mu, Q,
                   "", False, False, False)