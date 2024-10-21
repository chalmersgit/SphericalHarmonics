'''
MIT License

Copyright (c) 2018 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Spherical harmonics for radiance maps using numpy

Assumes:
Equirectangular format
theta: [0 to pi], from top to bottom row of pixels
phi: [0 to 2*Pi], from left to right column of pixels
'''

import numpy as np

# Spherical harmonics functions


import numpy as np

# Spherical Harmonics functions

def P(l, m, x):
    """
    Computes the associated Legendre polynomial P(l, m, x).
    
    Args:
        l (int): Band of the polynomial.
        m (int): Coefficient within band of the polynomial.
        x (ndarray): Input array.
    
    Returns:
        ndarray: Values of the Legendre polynomial.
    """
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm * np.ones(x.shape)

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if l == m + 1:
        return pmmp1

    pll = np.zeros(x.shape)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def divfact(a, b):
    """
    Computes the factorial ratio (a! / b!).

    Args:
        a (int): Numerator value.
        b (int): Denominator value.
    
    Returns:
        float: The factorial division result.
    """
    if b == 0:
        return 1.0
    v = 1.0
    x = a - b + 1.0
    while x <= a + b:
        v *= x
        x += 1.0
    return 1.0 / v


def factorial(x):
    """
    Computes the factorial of x.

    Args:
        x (int): Input integer.
    
    Returns:
        float: Factorial of x.
    """
    return 1.0 if x == 0 else x * factorial(x - 1)


def K(l, m):
    """
    Computes the normalization constant K(l, m).

    Args:
        l (int): Band.
        m (int): Coefficient within band.
    
    Returns:
        float: Normalization constant.
    """
    return np.sqrt(((2 * l + 1) * factorial(l - m)) / (4 * np.pi * factorial(l + m)))


def K_fast(l, m):
    """
    Computes a fast approximation of the normalization constant K(l, m).

    Args:
        l (int): Band.
        m (int): Coefficient within band.
    
    Returns:
        float: Approximation of the normalization constant.
    """
    cAM = abs(m)
    uVal = 1.0
    k = l + cAM
    while k > (l - cAM):
        uVal *= k
        k -= 1
    return np.sqrt((2.0 * l + 1.0) / (4 * np.pi * uVal))


def sh(l, m, theta, phi):
    """
    Computes the spherical harmonics function Y(l, m) for angles theta and phi.

    Args:
        l (int): Band.
        m (int): Coefficient within band.
        theta (ndarray): Azimuth angle.
        phi (ndarray): Elevation angle.
    
    Returns:
        ndarray: Spherical harmonics function value.
    """
    sqrt2 = np.sqrt(2.0)
    cos_theta = np.cos(theta)
    if m == 0:
        return K(l, m) * P(l, m, cos_theta)
    elif m > 0:
        return sqrt2 * K(l, m) * np.cos(m * phi) * P(l, m, cos_theta)
    else:
        return sqrt2 * K(l, -m) * np.sin(-m * phi) * P(l, -m, cos_theta)


def sh_terms(l_max):
    """
    Computes the total number of spherical harmonics terms for a given maximum bands.

    Args:
        l_max (int): Maximum bands.
    
    Returns:
        int: Number of spherical harmonics terms.
    """
    return (l_max + 1) * (l_max + 1)


def sh_index(l, m):
    """
    Computes the index for accessing the (l, m) spherical harmonics term in a 1D array.

    Args:
        l (int): Band.
        m (int): Coefficient within band.
    
    Returns:
        int: Index of the spherical harmonics term.
    """
    return l * l + l + m


def sh_evaluate(theta, phi, l_max):
    """
    Evaluates the spherical harmonics up to a given maximum bands for input angles.

    Args:
        theta (ndarray): Azimuth angles.
        phi (ndarray): Elevation angles.
        l_max (int): Maximum bands for the spherical harmonics.
    
    Returns:
        ndarray: Spherical harmonics coefficients matrix.
    """
    coeffs_matrix = np.zeros((theta.shape[0], phi.shape[0], sh_terms(l_max)))

    for l in range(0, l_max + 1):
        for m in range(-l, l + 1):
            index = sh_index(l, m)
            coeffs_matrix[:, :, index] = sh(l, m, theta, phi)

    return coeffs_matrix