#!/usr/bin/env python
# coding: utf-8
# ---------------------------------#
"""
File : test.py
Author : Ruello V., Tirel A.
Description : 3ième TD algorithme numerique
"""

from qr_transform import *
from householder import *
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------#
# Test Householder
# ---------------------------------#
#
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
		print "test_QR_transform : FAIL"


# def test_QR_transforms(n, m):
# 	U = create_orthogonal_matrix(n)
# 	V = create_orthogonal_matrix(m)

# ---------------------------------#
if __name__ == '__main__':
	test_householder()
	test_QR_transform(5)


# ---------------------------------#
