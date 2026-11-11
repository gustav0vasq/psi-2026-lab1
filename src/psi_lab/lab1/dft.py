"""Parte comun 2: DFT, resolucion y fuga espectral."""

import numpy as np


def dft(x):
    """DFT por definicion, O(N^2): X[k] = sum_n x[n] exp(-2j pi k n / N)."""
    x = np.asarray(x) # acepta listas y las convierte en arreglo
    N = x.size
    n = np.arange(N) # fila
    k = n[:, None] # columna
    W = np.exp(-2j * np.pi * k * n / N) # matriz N x N
    return W @ x # producto matriz-vector

def idft(X):
    """DFT inversa por definicion."""
    raise NotImplementedError

def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado. Retorna (freqs, amp)."""
    raise NotImplementedError