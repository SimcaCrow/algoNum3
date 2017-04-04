#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
# ---------------------------------#
"""
File : image.py
Author : Ruello V.
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

# ---------------------------------#

def load_picture(path):
    return  mpi.imread(path)


# ---------------------------------#

"""
Fonction calculant le SVD de la matrice
Pour l'instant, elle utilise numpy.
Elle devra utiliser la méthode implémentée dans le
TP.
"""
def svd(matrix):
    # print matrix
    # return algorithm(matrix, 500)
    return np.linalg.svd(matrix, full_matrices=True)


# ---------------------------------#

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


# ---------------------------------#

"""
Renvoie la matrice reconsitutée à l'aide du SVD et en gardant les k premières
valeurs singulières
"""
def matrix_svd(matrix, k):
    (U, s, V) = svd(matrix)
    reconstimg = np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(V[:k, :])

    #Gestion des dépassements d'intensité de couleur
    for i in xrange(reconstimg.shape[0]):
        for j in xrange(reconstimg.shape[1]):
            if reconstimg[i,j] > 1:
                reconstimg[i,j] = 1
            elif reconstimg[i,j] < 0:
                reconstimg[i,j] = 0

    return reconstimg


def matrix_svd_eighen_values(matrix):
    (U, s, V) = svd(matrix)
    return s;


# ---------------------------------#

"""
Fonction calculant la taille de l'image compressée en gardant
les k premières valeurs singulières
"""
def compute_compressed_size(img, k):
    compressedSize = 0
    compressedSize += img.shape[0] * k
    compressedSize += k
    compressedSize += k * img.shape[1]
    return compressedSize * 3


# ---------------------------------#

"""
Fonction calculant la taille de l'image source
"""
def compute_source_image_size(img):
    return img.shape[0] * img.shape[1] * img.shape[2]


# ---------------------------------#

"""
Fonction qui prend en paramètre l'image "src".
Elle applique la compression, affiche et renvoie l'image décompressée.
"""
def svd_compress(src, k, plot = True):
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
    if plot:
        img_plot(img)
    else:
        print "."
    return img


"""
Fonction qui affiche un graphe représentant la taille de l'image compressée en Fonction
du nombre de valeurs singulières gardées.
"""
def graph_size(img, start = 1, end = 200, step = 1):
    X = range(start, end, step)
    Y = list()
    Z = [compute_source_image_size(img)] *  ((end-start) / step)
    for k in range(start, end, step):
        Y.append(compute_compressed_size(img, k))
    plt.title(u"Taille de l'image compressée en fonction du rang")
    plt.ylabel(u"Taille de l'image compressée au rang k")
    plt.xlabel("k")
    plt.plot(X, Y)
    plt.plot(X, Z)
    plt.show()


    plt.ylabel(u"Taille de l'image compressée (octets)")
    plt.xlabel("Rang k")
    plt.plot(X, Y, label="Image compressée")
    plt.plot(X, Z, label="Image originale")
    plt.legend(loc = "bottom right")
    plt.show()

# ---------------------------------#


def graph_size_efficienty(img, start = 1, end = 500, step = 1):
    X = range(start, end, step)
    Y = list()
    srcSize = compute_source_image_size(img)
    for k in range(start, end, step):
        Y.append(compute_compressed_size(img, k) * 1.0 / srcSize)
    plt.title(u"Rapport de la taille de l'image compressée sur la taille de l'image d'origine en fonction du rang")
    plt.ylabel(u"(taille de l'image compressée) / (taille de l'image d'origine)")
    plt.xlabel("Rang k")
    plt.show()


# ---------------------------------#

def graph_eighen_values(img):
    (r, g, b) = get_rgb_matrices(img)

    rs = matrix_svd_eighen_values(r)
    gs = matrix_svd_eighen_values(g)
    bs = matrix_svd_eighen_values(b)

    X = range(0, 50)
    plt.plot(X, rs[:50], color='r')
    plt.plot(X, gs[:50], color='g')
    plt.plot(X, bs[:50], color='b')

    plt.title(u"Valeur des 50 premières valeurs singulières des matrices r, g, et b")
    plt.show()


# ---------------------------------#


def graph_eighen_values(img):
    (r, g, b) = get_rgb_matrices(img)

    rs = matrix_svd_eighen_values(r)
    gs = matrix_svd_eighen_values(g)
    bs = matrix_svd_eighen_values(b)

    X = range(0, 50)
    plt.plot(X, rs[:50], color='r', label="Composante rouge")
    plt.plot(X, gs[:50], color='g', label="Composante verte")
    plt.plot(X, bs[:50], color='b', label="Composante bleue")

    plt.title(u"Valeur des 50 premières valeurs singulières des matrices r, g et b")
    plt.ylabel(u"Valeur singulière")
    plt.xlabel("Indice")
    plt.legend(loc = "upper right")
    plt.show()

# ---------------------------------#

def graph_error_algebric(src, start = 1, end = 50, step = 5):
    X = range(start, end, step)
    R = list()
    G = list()
    B = list()
    for k in range(start, end, step):
        comp = svd_compress(src, k, False)
        r = 0
        g = 0
        b = 0
        for i in xrange(src.shape[0]):
            for j in xrange(src.shape[1]):
                r += (src[i,j,0] - comp[i,j,0])**2
                g += (src[i,j,1] - comp[i,j,1])**2
                b += (src[i,j,2] - comp[i,j,2])**2
        R.append(r)
        G.append(g)
        B.append(b)

    plt.ylabel(u"Erreur algébrique selon les 3 composantes")
    plt.xlabel("k")
    plt.plot(X, R, color='r')
    plt.plot(X, G, color='g')
    plt.plot(X, B, color='b')

    plt.title(u"Erreur algébrique selon les 3 composantes en fonction du rang")
    plt.ylabel(u"Erreur algébrique")
    plt.xlabel("Rang k")
    plt.plot(X, R, color='r', label="Composante rouge")
    plt.plot(X, G, color='g', label="Composante verte")
    plt.plot(X, B, color='b', label="Composante bleue")
    plt.legend(loc = "upper right")
    plt.show()

# ---------------------------------#
if __name__ == '__main__':
    k = 5
    imageName = 'rocket.png'

    imgSrc = load_picture(imageName)
    svd_compress(imgSrc, k)
    graph_size(imgSrc)
    graph_eighen_values(imgSrc)
    graph_error_algebric(imgSrc)

# ---------------------------------#
