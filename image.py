#!/usr/bin/python3
# ---------------------------------#
"""
File : image.py
Author : El-Habr C.
Description : 3ième TD algorithme numerique
"""
# ---------------------------------#

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpi

# ---------------------------------#

def img_plot():
    plt.figure('Earth')
    img1 = mpi.imread("earth.png")
    plt.imshow(img1)


    plt.figure('Rocket')
    img2 = mpi.imread("rocket.png")
    plt.imshow(img2)
    plt.show()


# ---------------------------------#
if __name__ == '__main__':

    img_plot()

# ---------------------------------#
