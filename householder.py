#!/usr/bin/env python
# coding: utf-8
import matplotlib as mp
import numpy as np
#     3 4 0
# 3
# 4
# 0
#         3
#         4
#         0
# 3 4 0

def householder_explicite(x, y):
    u = (x - y) * 1.0/norme_vecteur(x - y)
    n = len(u)
    M = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            M[i,j] = u[i] * u[j]
    H = np.eye(n) - 2 * M
    return H

def norme_vecteur(u):
    sum_tmp = 0
    n = len(u)
    for i in xrange(n):
        sum_tmp += u[i]**2
    return np.sqrt(sum_tmp)
