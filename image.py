#!/usr/bin/env python
# coding: utf-8
# ---------------------------------#
"""
File : image.py
Author : El-Habr C.
Description : 3ième TD algorithme numerique
"""
# ---------------------------------#

import numpy as np
from qr_transform import *
from householder import *
import matplotlib.pyplot as plt
import matplotlib.image as mpi


# ---------------------------------#

def img_plot(matrix):
    plt.imshow(matrix)
    plt.show()

"""
Renvoie les matrices (r, g, b) de l'image
"""
def get_image_matrices(path):
    rgb = mpi.imread(path)
    size = rgb.shape
    r = np.zeros((size[0],size[1]))
    g = np.zeros((size[0],size[1]))
    b = np.zeros((size[0],size[1]))
    for i in xrange(size[0]):
        for j in xrange(size[1]):
            r[i,j] = rgb[i,j,0]
            g[i,j] = rgb[i,j,1]
            b[i,j] = rgb[i,j,2]
    return (r, g, b)

def matrix_svd(matrix, k):
    (U, s, V) = np.linalg.svd(matrix, full_matrices=False)

    print U.shape, s.shape, V.shape
    reconstimg = np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])
    return reconstimg

def svd_compress(k):
    (r, g, b) = get_image_matrices('rocket.png')
    print r
    rc = matrix_svd(r, k)
    gc = matrix_svd(g, k)
    bc = matrix_svd(b, k)
    img = np.zeros((rc.shape[0],rc.shape[1], 3))
    for i in xrange(rc.shape[0]):
        for j in xrange(rc.shape[1]):
            img[i,j,0] = rc[i,j]
            img[i,j,1] = gc[i,j]
            img[i,j,2] = bc[i,j]
    img_plot(img)
# ---------------------------------#
if __name__ == '__main__':
    svd_compress(200)

# ---------------------------------#
