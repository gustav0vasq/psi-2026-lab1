"""Parte comun 3: convolucion."""

import numpy as np


def convolucion_directa(x, h):
    """Convolucion lineal (largo len(x) + len(h) - 1) por definicion"""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.zeros(len(x) + len(h) - 1)
    for i in range(len(x) + len(h) - 1):
        for j in range(len(h)):
            # x[i-j] solo existe si el indice cae dentro de x
            if i - j >= 0 and i - j < len(x):
                y[i] += h[j] * x[i - j]
    return y


def _fft(x, signo):
    """FFT radix-2 iterativa (Cooley-Tukey, decimacion en el tiempo) de largo potencia de 2.

    Calcula sum_n x[n] exp(signo 2j pi k n / N) en O(N log N): permuta x en orden de bits
    invertidos y combina pares de bloques de largo m en bloques de 2m con la mariposa
    (a + w b, a - w b), w = exp(signo j pi k / m), k = 0..m-1.
    """
    x = np.asarray(x, dtype=complex)
    N = x.size
    bits = N.bit_length() - 1
    if N != 1 << bits:
        raise ValueError(f"el largo debe ser potencia de 2, no {N}")
    n = np.arange(N)
    invertidos = np.zeros(N, dtype=int)
    for b in range(bits):
        invertidos |= ((n >> b) & 1) << (bits - 1 - b)
    X = x[invertidos]
    m = 1
    while m < N:
        w = np.exp(signo * 1j * np.pi * np.arange(m) / m)
        X = X.reshape(-1, 2 * m)
        a, b = X[:, :m], X[:, m:] * w
        X = np.concatenate([a + b, a - b], axis=1)
        m *= 2
    return X.reshape(N)


def fft_radix2(x):
    """FFT por el algoritmo de Cooley-Tukey radix-2: igual que dft(x) pero O(N log N).

    x: 1D de largo N potencia de 2 (ValueError si no). Retorna X complejo 1D de largo N.
    """
    return _fft(x, -1)


def ifft_radix2(X):
    """FFT inversa radix-2, con el factor 1/N. X: 1D de largo potencia de 2."""
    X = np.asarray(X)
    return _fft(X, +1) / X.size


def convolucion_fft(x, h):
    """Misma convolucion lineal, via el teorema de la convolucion (FFT radix-2 propia)"""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    largo = len(x) + len(h) - 1
    # Se rellena con ceros hasta una potencia de 2 >= largo: con menos, sale la circular
    N = 1 << (largo - 1).bit_length()
    x_rell = np.concatenate([x, np.zeros(N - len(x))])
    h_rell = np.concatenate([h, np.zeros(N - len(h))])
    y = ifft_radix2(fft_radix2(x_rell) * fft_radix2(h_rell))
    # Solo la parte real, y sin el relleno sobrante
    return y.real[:largo]


def convolucion_separable_2d(img, h_filas, h_cols):
    """Convolucion 2D completa con un nucleo separable"""
    img = np.asarray(img, dtype=float)
    h_filas = np.asarray(h_filas, dtype=float)
    h_cols = np.asarray(h_cols, dtype=float)
    alto, ancho = img.shape

    intermedio = np.zeros((alto, ancho + len(h_filas) - 1))
    for i in range(alto):
        intermedio[i, :] = convolucion_directa(img[i, :], h_filas)

    salida = np.zeros((alto + len(h_cols) - 1, intermedio.shape[1]))
    for j in range(intermedio.shape[1]):
        salida[:, j] = convolucion_directa(intermedio[:, j], h_cols)
    return salida
