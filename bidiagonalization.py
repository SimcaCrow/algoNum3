#!/usr/bin/env python
# coding: utf-8
# ---------------------------------#
"""
File : test.py
Author : Tirel A.
Description : 3ième TD algorithme numerique
"""
# ---------------------------------#

import numpy as np

# ---------------------------------#

def householder_explicite(x, y):
    u = (x - y) * 1.0/norme_vecteur(x - y)
    n = len(u)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = u[i] * u[j]
    H = np.eye(n) - 2 * M
    return H

# ---------------------------------#

def norme_vecteur(u):
    sum_tmp = 0
    n = len(u)
    for i in range(n):
        sum_tmp += u[i]**2
    return np.sqrt(sum_tmp)

# ---------------------------------#

def bidiagonalize(M):
    """ Factorise la matrice M (n x m) selon Q_l x B x Q_r = M, de telle sorte que B soit bidiagonale.
    Retourne le triplet (Q_l, B, Q_r). """
    (n, m) = np.shape(M)
    Q_l = np.identity(n)
    Q_r = np.identity(m)
    B = M
    e1_n = np.identity(n)[:,0]
    e1_m = np.identity(m)[:,0]

    for i in range(min(n,m)):
        src = B[i:n,i]
        dst = np.linalg.norm(src)*e1_n[0:(n-i)]
        H1 = householder_explicite(src, dst)
        H1 = extend(H1, np.identity(n), i, i)
        Q_l = np.dot(Q_l, H1)
        B = np.dot(H1, B)

        if ((i < m-2)):
            src = B[i, i:m]
            dst = src[0] * e1_m[:(m-i)]
            dst[1] = np.sqrt(np.linalg.norm(src)**2 - dst[0]**2)

            H2 = householder_explicite(src,  dst)
            H2 = extend(H2, np.identity(m), i, i)
            Q_r = np.dot(H2, Q_r)
            B = np.dot(B, H2)

    return (Q_l, B, Q_r)

# ---------------------------------#

def extend(N, M, i0, j0):
    """ Copie une matrice N (n1 x n2) dans une matrice M (m1 x m2). Les éléments de M concernés sont donc remplacés par ceux de N. Le point (0, 0) de N est placé en (i0, j0) dans M. Retourne M.
    Préconditions :
      i0 + n1 - 1 < m1
      j0 + n2 - 1 < m2 """
    (n1, n2) = np.shape(N)
    (m1, m2) = np.shape(M)

    for i in range(n1):
        for j in range(n2):
            M[i0 + i, j0 + j] = N[i, j]

    return M

# ---------------------------------#
if __name__ == '__main__':
    M=np.arange(15).reshape(3,5)
    print (M)
    print (M.size)
    print (np.zeros(3)[2])
    (A,B,C)=bidiagonalize(M)
    print "test"
    print (A)
    print (B)
    print (C)
    print (np.dot(A,np.dot(B,C)))
# ---------------------------------#