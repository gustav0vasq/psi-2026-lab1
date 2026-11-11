"""Parte comun 2: DFT, resolucion y fuga espectral."""

import numpy as np

def _dft_por_definicion(x, signo):
    """sum_n x[n] exp(signo * 2j pi k n / N) para k = 0..N-1. signo = -1 o +1."""
    x = np.asarray(x)
    N = x.size
    n = np.arange(N)
    k = n[:, None]
    W = np.exp(signo * 2j * np.pi * k * n / N)
    return W @ x

def dft(x):
    """DFT por definicion, O(N^2): X[k] = sum_n x[n] exp(-2j pi k n / N)."""
    return _dft_por_definicion(x, -1)

def idft(X):
    """DFT inversa por definicion."""
    X = np.asarray(X)
    return _dft_por_definicion(X, +1) / X.size

def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado. Retorna (freqs, amp)."""
    raise NotImplementedError