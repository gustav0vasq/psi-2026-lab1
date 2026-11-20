"""Caso C, tarea C5: aliasing espacial"""

import numpy as np

from psi_lab.lab1.convolucion import convolucion_separable_2d
from psi_lab.lab1.filtros import fir_pasabajos


def zone_plate(M):
    """Zone plate de M x M """
    # centradas, asi en el borde se llega justo a 0.5 ciclos/px
    x = np.arange(M) - M / 2
    X, Y = np.meshgrid(x, x)
    return (1 + np.cos(np.pi * (X**2 + Y**2) / M)) / 2


def reducir(img, k):
    img = np.asarray(img, dtype=float)
    return img[::k, ::k]


def pasabajos_separable_2d(img, fc, n_coef):
    """Pasabajos 2D separable """
    img = np.asarray(img, dtype=float)
    # fs = 1 porque hay una muestra por pixel
    h = fir_pasabajos(fc, 1, n_coef)
    completa = convolucion_separable_2d(img, h, h)
    # se recorta el centro para que quede del tamano de img
    retardo = (n_coef - 1) // 2
    alto, ancho = img.shape
    return completa[retardo : retardo + alto, retardo : retardo + ancho]
