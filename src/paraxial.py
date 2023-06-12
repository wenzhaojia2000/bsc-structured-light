# -*- coding: utf-8 -*-
"""
Coder: James Jia
Description: Coding of paraxial equations given by _________
"""

import numpy as np
from scipy.misc import derivative
from rdf import radialDistributionFunction

multiplier = r'$\varepsilon_0^{-2}\Omega^2n$'

def e1e2Term(p:int, l:int, w0:float, k:float, sigma:int,\
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
    
    Optional keyword arguments:
     rfzseperate (bool): send e1e2 as an array with [rho,phi,z] values
     e1e2seperate (bool): send e1e2 as an array with [bar(e)1e2, e1bar(e)2] values

    e1e2 : E1E2 term of the matrix element
    '''
    # Get values from kwargs. The kwargs exist for compatibility in the
    # visualizeFunct function in FinalReport_Visualize file
    mu = kwargs["mu"]
    Q = kwargs["Q"]
    rfzseperate = False if kwargs.get("rfzseperate") is None\
        else kwargs.get("rfzseperate")
    e1e2seperate = False if kwargs.get("e1e2seperate") is None\
        else kwargs.get("e1e2seperate")
    
    def rdf(rho):
        return radialDistributionFunction(p, l, w0, rho)
    # Arrays need to be consistent, so "0" needs to have the same shape
    # as phi and rho.
    assert np.shape(phi) == np.shape(rho)
    arrayShape = np.shape(phi)
    zeroArray = np.zeros(arrayShape)
    
    # Calculations of values as follows
    derf = derivative(rdf, rho, dx=0.01)
    barQ = np.conjugate(Q)
    barmu = np.conjugate(mu)
    xpisy = np.array([1,1j*sigma,0])
    xmisy = np.array([1,-1j*sigma,0])
    
    rhohatTerm = rdf(rho)*derf *\
             np.array([np.cos(phi), np.sin(phi), zeroArray])
    phihatTerm = rdf(rho)**2/rho * (1j/l) *\
             np.array([-np.sin(phi), np.cos(phi), zeroArray])
    zhatTerm = 1j*k *\
           np.array([zeroArray, zeroArray, np.ones(arrayShape)])
    
    if rfzseperate == False and e1e2seperate == False:
        rfzarray1 = rhohatTerm + phihatTerm + zhatTerm
        rfzarray2 = rhohatTerm - phihatTerm - zhatTerm
        
        # Compute einsum. "rfzarray" is a rank 1 tensor, but its components (x,y,z)
        # are 2D arrays, which makes einsum think it is rank 3. So there needs to
        # be extraction of "rfzarray" into individual "rfz" terms, calculate each
        # einsum with a "rfz" term seperately, and reassemble them back into an
        # array.
        e1e2 = np.empty(arrayShape, dtype = 'complex_')
        for i in range(arrayShape[0]):
            for j in range(arrayShape[1]):
                rfz1 = [rfzarray1[0][i][j],rfzarray1[1][i][j],rfzarray1[2][i][j]]
                rfz2 = [rfzarray2[0][i][j],rfzarray2[1][i][j],rfzarray2[2][i][j]]
                e1e2[i][j] = np.einsum("i,k,k,ij,j",xpisy,xmisy,barmu,Q,rfz1)\
                           + np.einsum("i,k,i,kj,j",xpisy,xmisy,mu,barQ,rfz2)
    elif rfzseperate == True and e1e2seperate == False:
        rfzarray1 = [rhohatTerm, phihatTerm, zhatTerm]
        rfzarray2 = [rhohatTerm, -phihatTerm, -zhatTerm]
        
        # now has an additional argument for rho, phi, zhat
        e1e2 = np.empty((3,arrayShape[0],arrayShape[1]), dtype = 'complex_')
        for i in range(arrayShape[0]):
            for j in range(arrayShape[1]):
                rho1 = [rfzarray1[0][0][i][j],rfzarray1[0][1][i][j],rfzarray1[0][2][i][j]]
                rho2 = [rfzarray2[0][0][i][j],rfzarray2[0][1][i][j],rfzarray2[0][2][i][j]]
                phi1 = [rfzarray1[1][0][i][j],rfzarray1[1][1][i][j],rfzarray1[1][2][i][j]]
                phi2 = [rfzarray2[1][0][i][j],rfzarray2[1][1][i][j],rfzarray2[1][2][i][j]]
                z1 = [rfzarray1[2][0][i][j],rfzarray1[2][1][i][j],rfzarray1[2][2][i][j]]
                z2 = [rfzarray2[2][0][i][j],rfzarray2[2][1][i][j],rfzarray2[2][2][i][j]]
                
                e1e2[0][i][j] = np.einsum("i,k,k,ij,j",xpisy,xmisy,barmu,Q,rho1)\
                              + np.einsum("i,k,i,kj,j",xpisy,xmisy,mu,barQ,rho2)
                e1e2[1][i][j] = np.einsum("i,k,k,ij,j",xpisy,xmisy,barmu,Q,phi1)\
                              + np.einsum("i,k,i,kj,j",xpisy,xmisy,mu,barQ,phi2)
                e1e2[2][i][j] = np.einsum("i,k,k,ij,j",xpisy,xmisy,barmu,Q,z1)\
                              + np.einsum("i,k,i,kj,j",xpisy,xmisy,mu,barQ,z2)
    else:
        rfzarray1 = rhohatTerm + phihatTerm + zhatTerm
        rfzarray2 = rhohatTerm - phihatTerm - zhatTerm
        
        # has an additional argument for bar(e)1e2, e1bar(e)2
        e1e2 = np.empty((2,arrayShape[0],arrayShape[1]), dtype = 'complex_')
        for i in range(arrayShape[0]):
            for j in range(arrayShape[1]):
                rfz1 = [rfzarray1[0][i][j],rfzarray1[1][i][j],rfzarray1[2][i][j]]
                rfz2 = [rfzarray2[0][i][j],rfzarray2[1][i][j],rfzarray2[2][i][j]]
                e1e2[0][i][j] = np.einsum("i,k,k,ij,j",xpisy,xmisy,barmu,Q,rfz1)
                e1e2[1][i][j] = np.einsum("i,k,i,kj,j",xpisy,xmisy,mu,barQ,rfz2)
    
    return e1e2


def e2e2Term(p:int, l:int, w0:float, k:float, sigma:int,\
             rho:"np.array", phi:"np.array", **kwargs) -> float:
    '''
    p (int): Radial Index  
    l (int): Topological charge  
    w0 : Beam waist at z = 0, in m
    k : Wavenumber
    rho (2D array): rho position, in m
    phi (2D array): phi position, in rad
    e (opt., vect.): polarization operator
    
    Required keyword arguments:    
     Q (matr.): quadrupole transition moment
    
    Optional keyword arguments:
     rfzseperate (bool): send e1e2 as an array with [rho,phi,z] values

    e2e2 : E2E2 term of the matrix element
    '''
    # Get values from kwargs. The kwargs exist for compatibility in the
    # visualizeFunct function in FinalReport_Visualize file
    Q = kwargs["Q"]
    rfzseperate = False if kwargs.get("rfzseperate") is None\
        else kwargs.get("rfzseperate")

    def rdf(rho):
        return radialDistributionFunction(p, l, w0, rho)
    # Arrays need to be consistent, so "0" needs to have the same shape
    # as phi and rho.
    assert np.shape(phi) == np.shape(rho)
    arrayShape = np.shape(phi)
    zeroArray = np.zeros(arrayShape)
    
    # Calculations of values as follows
    derf = derivative(rdf, rho, dx=0.01)
    barQ = np.conjugate(Q)
    xpisy = np.array([1,1j*sigma,0])
    xmisy = np.array([1,-1j*sigma,0])
    
    rhohatTerm = rdf(rho)*derf *\
             np.array([np.cos(phi), np.sin(phi), zeroArray])
    phihatTerm = rdf(rho)**2/rho * (1j/l) *\
             np.array([-np.sin(phi), np.cos(phi), zeroArray])
    zhatTerm = 1j*k *\
           np.array([zeroArray, zeroArray, np.ones(arrayShape)])
           
    if rfzseperate == False:
        rfzarray1 = rhohatTerm + phihatTerm + zhatTerm
        rfzarray2 = rhohatTerm - phihatTerm - zhatTerm
        
        # Compute einsum. "rfzarray" is a rank 1 tensor, but its components (x,y,z)
        # are 2D arrays, which makes einsum think it is rank 3. So there needs to
        # be extraction of "rfzarray" into individual "rfz" terms, calculate each
        # einsum with a "rfz" term seperately, and reassemble them back into an
        # array.
        e2e2 = np.empty(arrayShape, dtype = 'complex_')
        for i in range(arrayShape[0]):
            for j in range(arrayShape[1]):
                rfz1 = [rfzarray1[0][i][j],rfzarray1[1][i][j],rfzarray1[2][i][j]]
                rfz2 = [rfzarray2[0][i][j],rfzarray2[1][i][j],rfzarray2[2][i][j]]
                e2e2[i][j] = np.einsum("i,k,ij,kl,j,l",xpisy,xmisy,Q,barQ,rfz1,rfz2)
    else:
        rfzarray1 = [rhohatTerm, phihatTerm, zhatTerm]
        rfzarray2 = [rhohatTerm, -phihatTerm, -zhatTerm]
        
        # now has an additional argument for rho, phi, zhat
        e2e2 = np.empty((3,arrayShape[0],arrayShape[1]), dtype = 'complex_')
        for i in range(arrayShape[0]):
            for j in range(arrayShape[1]):
                rho1 = [rfzarray1[0][0][i][j],rfzarray1[0][1][i][j],rfzarray1[0][2][i][j]]
                rho2 = [rfzarray2[0][0][i][j],rfzarray2[0][1][i][j],rfzarray2[0][2][i][j]]
                phi1 = [rfzarray1[1][0][i][j],rfzarray1[1][1][i][j],rfzarray1[1][2][i][j]]
                phi2 = [rfzarray2[1][0][i][j],rfzarray2[1][1][i][j],rfzarray2[1][2][i][j]]
                z1 = [rfzarray1[2][0][i][j],rfzarray1[2][1][i][j],rfzarray1[2][2][i][j]]
                z2 = [rfzarray2[2][0][i][j],rfzarray2[2][1][i][j],rfzarray2[2][2][i][j]]
                
                e2e2[0][i][j] = np.einsum("i,k,ij,kl,j,l",xpisy,xmisy,Q,barQ,rho1,rho2)
                e2e2[1][i][j] = np.einsum("i,k,ij,kl,j,l",xpisy,xmisy,Q,barQ,phi1,phi2)
                e2e2[2][i][j] = np.einsum("i,k,ij,kl,j,l",xpisy,xmisy,Q,barQ,z1,z2)

    return e2e2