#!/usr/bin/env python
# coding: utf-8
import numpy as np
from householder import *

"""Cette fonction renvoie un vecteur composé uniquement de 0 sauf à la ligne i"""
def vector_householder(i,n):
    A      = np.zeros(n)
    A[i]   = 1
    return A



def bidiagonalisation(A):
    n          = len(A)
    m          = len(A[0])
    Qleft      = np.eye(n)
    Qright     = np.eye(n)
    BD         = A
    for i in range (0,n-1):
        Q1           = householder_explicite(BD[i:n,i],vector_householder(0,n-i))
        Q11          = np.zeros(n)
        print Q1
        print Q11
#        for k in range (0,Q1.size):
 #           Q11[k]   = Q1[k]
            
        Qleft        = np.dot(Qleft,Q11)
        BD           = np.dot(Q11,BD)
        if i!=(m-2):
            Q2       = householder_explicite(BD[i,(i+1):m],vector_householder(0,m-1-i))
            Q21      = np.zeros(m)
            for k in range (0,Q2.size):
                Q21[k] = Q2[k]
            Qright   = np.dot(Q21,Qright)
            BD       = np.dot(BD,Q21)
    return(Qleft, BD, Qright)

M=np.arange(9).reshape(3,3)
print M
print (M.size)
print (np.zeros(3)[2])
(A,B,C)=bidiagonalisation(M)
print A
print B
print C
print np.dot(A,np.dot(B,C))
