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
    """Espectro de amplitud de un lado. 
    x: 1D real de largo N; fs [Hz]; ventana en {"rect", "hann", "hamming"};
    nfft >= N rellena con ceros (por defecto nfft = N).
    Retorna (freqs, amp), ambos de largo nfft // 2 + 1, con freqs en [0, fs/2] Hz.
    Un tono A cos(2 pi f t) que cae justo en un bin da un pico de altura A.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if nfft is None:
        nfft = N
    if nfft < N:
        raise ValueError(f"nfft ({nfft}) debe ser >= len(x) ({N})")

    # a) ventana   b) relleno con ceros
    w = _ventana(ventana, N)
    relleno = np.zeros(nfft)
    relleno[:N] = x * w

    # c) DFT y solo la mitad positiva
    X = dft(relleno)[: nfft // 2 + 1]

    # d) normalizar: dividir por sum(w) y duplicar todo salvo DC (y Nyquist si nfft es par)
    amp = np.abs(X) / np.sum(w)
    if nfft % 2 == 0:
        amp[1:-1] *= 2
    else:
        amp[1:] *= 2

    freqs = np.arange(nfft // 2 + 1) * fs / nfft
    return freqs, amp

_VENTANAS = ("rect", "hann", "hamming")
def _ventana(nombre, N):
    """Ventana de N muestras (misma definicion que np.hanning y np.hamming)."""
    if nombre not in _VENTANAS:
        raise ValueError(f"ventana debe ser una de {_VENTANAS}, no {nombre!r}")
    if nombre == "rect" or N == 1:
        return np.ones(N)
    n = np.arange(N)
    coseno = np.cos(2 * np.pi * n / (N - 1))
    if nombre == "hann":
        return 0.5 - 0.5 * coseno
    return 0.54 - 0.46 * coseno