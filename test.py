#!/usr/bin/env python
# coding: utf-8
# ---------------------------------#
"""
File : test.py
Author : Ruello V., Tirel A.
Description : 3ième TD algorithme numerique
"""
# ---------------------------------#

from qr_transform import *
from householder import *
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------#
# Test Householder
# ---------------------------------#

def test_householder():
    x = np.array([3., 4., 0.])
    y = np.array([0.,0.,5.])
    H = householder_explicite(x, y)
    H_exp = np.array([ [0.64, -0.48, 0.6], [-0.48, 0.36, 0.8], [0.6, 0.8, 0]])
    print "Résultat attendu : ", H_exp
    print "Résultat obtenu : ", H
    print "Résultat attendu : ",  np.dot(H_exp,x)
    print "Résultat obtenu : ",  np.dot(H,x)

# ---------------------------------#
# Test transformations QR
# ---------------------------------#

def test_QR_transform(n):
    Q = create_orthogonal_matrix(n)
    R = create_triang_sup_matrix(n)
    A = np.dot(Q, R)
    (Qt, Rt) = np.linalg.qr(A)
    if np.array_equal(A,np.dot(Qt, Rt)):
        print "test_QR_transform : OK"
    else:
        print "test_QR_transform : INCERTAIN"
        print "Les deux matrices suivantes doivent être égales : "
        print A
        print np.dot(Qt, Rt)

# ---------------------------------#

def test_QR_transforms(n, Nmax):
    BD = create_random_matrix(n)
    (U, S, V) = algorithm(BD, Nmax)
    print S
    print somme_abs_valeurs_extra_diagonales(S)
    res = np.dot(np.dot(U, S), V)
    if np.array_equal(BD, res):
        print "test_QR_transforms : OK"
    else:
        print "test_QR_transforms : INCERTAIN"
        print "Les deux matrices suivantes doivent être égales : "
        print BD
        print res

# ---------------------------------#

def test_convergence(n):
    BD = create_random_matrix(n)
    Ns = range(0, 15, 1)
    values = list()
    for nmax in Ns:
        (U, S, V) = algorithm(BD, nmax)
        values.append(somme_abs_valeurs_extra_diagonales(S))
    #print values
    #plt.plot(Ns, values)
    #plt.show()

# ---------------------------------#
if __name__ == '__main__':

    test_householder()
    test_QR_transform(5)
    test_QR_transforms(5, 1000)
    test_convergence(5)

# ---------------------------------#
