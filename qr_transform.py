#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random

# U = Id; V = Id; S = BD;
# For i from 0 to NMax
#   (Q1, R1) = decomp_qr(matrix_transpose(S))
#   (Q2, R2) = decomp_qr(matrix_transpose(R1))
#   S = R2;
#   U = U * Q2;
#   V = matrix_transpose(Q1) * V
# End for
# Return (U,S,V)

"""
Fonction générant de manière aléatoire une matrice orthogonale
Pour les générer on choisit de générer des matrices de permutation aléatoires
"""
def create_orthogonal_matrix(n):
    A = np.zeros((n,n))
    bools = [False] * n;
    for i in xrange(n):
        rand_indice = random.randint(0, n-1)
        while bools[rand_indice]:
            rand_indice = random.randint(0, n-1)
        A[i,rand_indice] = 1
        bools[rand_indice] = True
    return A

"""
Fonction permettant de générer une matrice triangualire supérieure aléatoire
"""
def create_triang_sup_matrix(n):
    A = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(i, n):
            A[i,j] = random.randint(0, n) #On prend n, ça peut changer. Pas de raison précise
    return A


"""
Fonction permettant de générer une matrice diagonale aléatoire dont la suite
des coefficients diagonaux est décroissante
"""
def create_diag_matrix(n):
    A = np.zeros((n,n))
    prec = n+1 #prec doit être au plus supérieur à n
    for i in xrange(n):
        A[i,i] = random.randint(0, prec)
        prec = A[i,i]
    return A

def algorithm(A, NMax):
    n = len(A)
    U = np.eyes(n)
    V = np.eyes(n)
    S = A
    for i in xrange(NMax):
        (Q1, R1) = np.linalg.qr(S)
        (Q2, R2) = np.linalg.qr(R1)
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.transpose(Q1), V)
    return (U, S, V)
