# -*- coding: utf-8 -*-
"""
Coder: James Jia
Description: Coding of Non-paraxial equations given by _________
"""

import numpy as np
from scipy.misc import derivative
from rdf import radialDistributionFunction

multiplier = r'$-\frac{1}{2}\Omega^2n$'

def e1e2Term(p:int, l:int, w0:float, k:float, sigma:float,\
             rho:"np.array", phi:"np.array", **kwargs) -> float:
    '''
    p (int): Radial Index  
    l (int): Topological charge  
    w0 : Beam waist at z = 0, in m
    k : Wavenumber
    sigma : polarization
    rho (2D array): rho position, in m
    phi (2D array): phi position, in rad
    
    Required keyword arguments: 
     mu (vect.): electric dipole transition moment
     Q (matr.): quadrupole transition moment

    e1e2 : E1E2 term of the matrix element
    '''
    # Get values from kwargs. The kwargs exist for compatibility in the
    # visualizeFunct function in FinalReport_Visualize file
    mu = kwargs["mu"]
    Q = kwargs["Q"]

    def rdf(rho):
        return radialDistributionFunction(p, l, w0, rho)
    def rdfOverr(rho):
        return radialDistributionFunction(p, l, w0, rho)/rho
    # Arrays need to be consistent, so "0" needs to have the same shape
    # as phi and rho.
    assert np.shape(phi) == np.shape(rho)
    arrayShape = np.shape(phi)
    zeroArray = np.zeros(arrayShape)
    
    # some terms defined below
    derf = derivative(rdf, rho, dx=0.01)
    dderf = derivative(rdf, rho, dx=0.01, n=2)
    barmu = np.conjugate(mu)
    barQ = np.conjugate(Q)
    xhat = np.array([np.ones(arrayShape), zeroArray, zeroArray])
    yhat = np.array([zeroArray, np.ones(arrayShape), zeroArray])
    zhat = np.array([zeroArray, zeroArray, np.ones(arrayShape)])
    rhohat = np.array([np.cos(phi), np.sin(phi), zeroArray])
    phihat = np.array([-np.sin(phi), np.cos(phi), zeroArray])
    xpisyarray = xhat + 1j*sigma*yhat
    xmisyarray = xhat - 1j*sigma*yhat
    
    # start calculation of the first half of the equation ({a1}{a2})
    a1array = xpisyarray*rdf(rho) + (1j/k)*(derf - (l*sigma/rho)*rdf(rho))*\
         np.e**(1j*sigma*phi)*zhat
    a2p1array = rhohat*derf - (1j*l/rho)*rdf(rho)*phihat - 1j*rdf(rho)*k*zhat
    a2p2array = (-rhohat*(1j/k)*dderf) -\
           (-rhohat*(1j/k)*l*sigma*derivative(rdfOverr,rho,dx=0.01)) -\
           ((l+sigma)/(k*rho))*phihat*(derf-((l*sigma)/rho)*rdf(rho)) -\
           zhat*(derf-((l*sigma)/rho)*rdf(rho))
    a2p3array = np.e**(-1j*sigma*phi)*zhat
    
    # a2 is a rank-two tensor with (x, y) values which internally makes it a rank
    # 4 tensor, so the shape of the array will be ([no. of x values],[no. of y
    # values], 3, 3).
    a2array = np.empty((arrayShape[0],arrayShape[1],3,3), dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            a2p1 = [a2p1array[0][i][j],a2p1array[1][i][j],a2p1array[2][i][j]]
            a2p2 = [a2p2array[0][i][j],a2p2array[1][i][j],a2p2array[2][i][j]]
            a2p3 = [a2p3array[0][i][j],a2p3array[1][i][j],a2p3array[2][i][j]]
            xmisy = [xmisyarray[0][i][j],xmisyarray[1][i][j],xmisyarray[2][i][j]]
            a2array[i][j] = np.einsum('l,k',a2p1,xmisy) + np.einsum('l,k',a2p2,a2p3)
    
    a = np.empty(arrayShape, dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            a1 = [a1array[0][i][j],a1array[1][i][j],a1array[2][i][j]]
            # the (0,1,2) values are at the end since the shape of a2 is also
            # with the (...3, 3) at the end.
            a2 = [a2array[i][j][0],a2array[i][j][1],a2array[i][j][2]]
            a[i][j] = np.einsum('i,lk,i,kl',a1,a2,mu,barQ)

    # start calculation of the second half of the equation ({b1}{b2})
    b1array = xmisyarray*rdf(rho) - (1j/k)*(derf - (l*sigma/rho)*rdf(rho))*\
         np.e**(-1j*sigma*phi)*zhat
    b2p1array = rhohat*derf + (1j*l/rho)*rdf(rho)*phihat + 1j*rdf(rho)*k*zhat
    b2p2array = (rhohat*(1j/k)*dderf) -\
           (rhohat*(1j/k)*l*sigma*derivative(rdfOverr,rho,dx=0.01)) -\
           ((l+sigma)/(k*rho))*phihat*(derf-((l*sigma)/rho)*rdf(rho)) -\
           zhat*(derf-((l*sigma)/rho)*rdf(rho))
    b2p3array = np.e**(1j*sigma*phi)*zhat
    
    # similarly with the above.
    b2array = np.empty((arrayShape[0],arrayShape[1],3,3), dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            b2p1 = [b2p1array[0][i][j],b2p1array[1][i][j],b2p1array[2][i][j]]
            b2p2 = [b2p2array[0][i][j],b2p2array[1][i][j],b2p2array[2][i][j]]
            b2p3 = [b2p3array[0][i][j],b2p3array[1][i][j],b2p3array[2][i][j]]
            xpisy = [xpisyarray[0][i][j],xpisyarray[1][i][j],xpisyarray[2][i][j]]
            b2array[i][j] = np.einsum('j,i',b2p1,xpisy) + np.einsum('j,i',b2p2,b2p3)
    
    b = np.empty(arrayShape, dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            b1 = [b1array[0][i][j],b1array[1][i][j],b1array[2][i][j]]
            # the (0,1,2) values are at the end since the shape of a2 is also
            # with the (...3, 3) at the end.
            b2 = [b2array[i][j][0],b2array[i][j][1],b2array[i][j][2]]
            b[i][j] = np.einsum('k,ji,k,ij',b1,b2,barmu,Q)
    
    return a + b

def e2e2Term(p:int, l:int, w0:float, k:float, sigma:float,\
             rho:"np.array", phi:"np.array", **kwargs) -> float:
    '''
    p (int): Radial Index  
    l (int): Topological charge  
    w0 : Beam waist at z = 0, in m
    k : Wavenumber
    sigma : polarization
    rho (2D array): rho position, in m
    phi (2D array): phi position, in rad
    
    Required keyword arguments:
     Q (matr.): quadrupole transition moment

    e1e2 : E1E2 term of the matrix element
    '''
    
    # Get values from kwargs. The kwargs exist for compatibility in the
    # visualizeFunct function in FinalReport_Visualize file
    Q = kwargs["Q"]
    
    def rdf(rho):
        return radialDistributionFunction(p, l, w0, rho)
    def rdfOverr(rho):
        return radialDistributionFunction(p, l, w0, rho)/rho
    # Arrays need to be consistent, so "0" needs to have the same shape
    # as phi and rho.
    assert np.shape(phi) == np.shape(rho)
    arrayShape = np.shape(phi)
    zeroArray = np.zeros(arrayShape)
    
    # some terms defined below
    derf = derivative(rdf, rho, dx=0.01)
    dderf = derivative(rdf, rho, dx=0.01, n=2)
    barQ = np.conjugate(Q)
    xhat = np.array([np.ones(arrayShape), zeroArray, zeroArray])
    yhat = np.array([zeroArray, np.ones(arrayShape), zeroArray])
    zhat = np.array([zeroArray, zeroArray, np.ones(arrayShape)])
    rhohat = np.array([np.cos(phi), np.sin(phi), zeroArray])
    phihat = np.array([-np.sin(phi), np.cos(phi), zeroArray])
    xpisyarray = xhat + 1j*sigma*yhat
    xmisyarray = xhat - 1j*sigma*yhat
    
    # start calculation...
    b2p1array = rhohat*derf + (1j*l/rho)*rdf(rho)*phihat + 1j*rdf(rho)*k*zhat
    b2p2array = (rhohat*(1j/k)*dderf) -\
           (rhohat*(1j/k)*l*sigma*derivative(rdfOverr,rho,dx=0.01)) -\
           ((l+sigma)/(k*rho))*phihat*(derf-((l*sigma)/rho)*rdf(rho)) -\
           zhat*(derf-((l*sigma)/rho)*rdf(rho))
    b2p3array = np.e**(1j*sigma*phi)*zhat
    
    b2array = np.empty((arrayShape[0],arrayShape[1],3,3), dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            b2p1 = [b2p1array[0][i][j],b2p1array[1][i][j],b2p1array[2][i][j]]
            b2p2 = [b2p2array[0][i][j],b2p2array[1][i][j],b2p2array[2][i][j]]
            b2p3 = [b2p3array[0][i][j],b2p3array[1][i][j],b2p3array[2][i][j]]
            xpisy = [xpisyarray[0][i][j],xpisyarray[1][i][j],xpisyarray[2][i][j]]
            b2array[i][j] = np.einsum('j,i',b2p1,xpisy) + np.einsum('j,i',b2p2,b2p3)
    
    a2p1array = rhohat*derf - (1j*l/rho)*rdf(rho)*phihat - 1j*rdf(rho)*k*zhat
    a2p2array = (-rhohat*(1j/k)*dderf) -\
           (-rhohat*(1j/k)*l*sigma*derivative(rdfOverr,rho,dx=0.01)) -\
           ((l+sigma)/(k*rho))*phihat*(derf-((l*sigma)/rho)*rdf(rho)) -\
           zhat*(derf-((l*sigma)/rho)*rdf(rho))
    a2p3array = np.e**(-1j*sigma*phi)*zhat
    
    a2array = np.empty((arrayShape[0],arrayShape[1],3,3), dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            a2p1 = [a2p1array[0][i][j],a2p1array[1][i][j],a2p1array[2][i][j]]
            a2p2 = [a2p2array[0][i][j],a2p2array[1][i][j],a2p2array[2][i][j]]
            a2p3 = [a2p3array[0][i][j],a2p3array[1][i][j],a2p3array[2][i][j]]
            xmisy = [xmisyarray[0][i][j],xmisyarray[1][i][j],xmisyarray[2][i][j]]
            a2array[i][j] = np.einsum('l,k',a2p1,xmisy) + np.einsum('l,k',a2p2,a2p3)
    
    ba = np.empty(arrayShape, dtype = 'complex_')
    for i in range(arrayShape[0]):
        for j in range(arrayShape[1]):
            b2 = [b2array[i][j][0],b2array[i][j][1],b2array[i][j][2]]
            a2 = [a2array[i][j][0],a2array[i][j][1],a2array[i][j][2]]
            ba[i][j] = np.einsum('ji,lk,ij,kl',b2,a2,Q,barQ)
    return ba