#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
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
import matplotlib


# ---------------------------------#

def img_plot(matrix):
    plt.imshow(matrix)
    plt.show()

def load_picture(path):
    return  mpi.imread(path)


"""
Renvoie les matrices (r, g, b) de l'image
"""
def get_rgb_matrices(rgb):
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
    (U, s, V) = np.linalg.svd(matrix, full_matrices=True)
    #print U.shape, s.shape, V.shape
    reconstimg = np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])
    # print '-----'
    # ucalc = 1
    # scalc = 1
    # vcalc = 1
    # for x in U[:, :k].shape:
    #     ucalc *= x
    # for x in s[:k].shape:
    #     scalc *= x
    # for x in V[:k, :].shape:
    #     vcalc *= x
    # print "Taille = ", ucalc + scalc + vcalc
    #
    # print np.matrix(U[:, :k]).shape
    # print np.diag(s[:k]).shape
    # print np.matrix(V[:k, :]).shape
    # print reconstimg.shape
    for i in xrange(reconstimg.shape[0]):
        for j in xrange(reconstimg.shape[1]):
            if reconstimg[i,j] > 1:
                reconstimg[i,j] = 1
            elif reconstimg[i,j] < 0:
                reconstimg[i,j] = 0
    return reconstimg

def compute_compressed_size(img, k):
    compressedSize = 0
    compressedSize += img.shape[0] * k
    compressedSize += k
    compressedSize += k * img.shape[1]
    return compressedSize * 3

def compute_source_image_size(img):
    return img.shape[0] * img.shape[1] * img.shape[2]


def svd_compress(src, k):
    (r, g, b) = get_rgb_matrices(src)
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
    return img


def graph_size(img, start = 0, end = 200, step = 1):
    X = range(start, end, step)
    Y = list()
    Z = [compute_source_image_size(img)] *  ((end-start) / step)
    for k in range(start, end, step):
        Y.append(compute_compressed_size(img, k))
    plt.title(u"Graphique représentant la taille de l'image compressée en fonction du rang")
    plt.ylabel(u"Taille de l'image compressée au rang k")
    plt.xlabel("k")
    plt.plot(X, Y)
    plt.plot(X, Z)
    plt.show()

def graph_size_efficienty(img, start = 0, end = 500, step = 1):
    X = range(start, end, step)
    Y = list()
    srcSize = compute_source_image_size(img)
    for k in range(start, end, step):
        Y.append(compute_compressed_size(img, k) * 1.0 / srcSize)
    plt.title(u"Graphique représentant le rapport entre la taille de l'image compressée et la taille de l'image d'origine en fonction du rang")
    plt.ylabel(u"Rapport entre la taille de l'image compressée et la taille de l'image d'origine au rang k")
    plt.xlabel("k")
    plt.plot(X, Y)
    plt.show()

def graph_error_algebric(img, start = 0, end = 500, step = 1):
    X = range(start, end, step)
    Y = list()

    for k in range(start, end, step):
        Y.append(compute_compressed_size(img, k) * 1.0 / srcSize)
    plt.title(u"Graphique représentant le rapport entre la taille de l'image compressée et la taille de l'image d'origine en fonction du rang")
    plt.ylabel(u"Rapport entre la taille de l'image compressée et la taille de l'image d'origine au rang k")
    plt.xlabel("k")
    plt.plot(X, Y)
    plt.show()

# ---------------------------------#
if __name__ == '__main__':
    k = 5
    imageName = 'rocket.png'

    imgSrc = load_picture(imageName)
    svd_compress(imgSrc, k)
    graph_size(imgSrc)
    graph_size_efficienty(imgSrc)


# ---------------------------------#
