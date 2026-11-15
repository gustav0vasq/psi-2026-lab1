"""Parte comun 3: convolucion."""
import numpy as np

from psi_lab.lab1.dft import dft, idft


def convolucion_directa(x, h):
    """Convolucion lineal (largo len(x) + len(h) - 1) por definicion"""
    x=np.asarray(x, dtype=float)
    h=np.asarray(h, dtype=float)
    y=np.zeros(len(x) + len(h) -1)
    for i in range(len(x) + len(h) -1):
        for j in range(len(h)):
            # x[i-j] solo existe si el indice cae dentro de x
            if i - j >= 0 and i - j < len(x):
                y[i] += h[j] * x[i - j]
    return y
        
def convolucion_fft(x, h):
    """Misma convolucion lineal, via el teorema de la convolucion"""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    # se rellena con ceros hasta L si no sale la convolucion circular
    x_rell = np.concatenate([x, np.zeros(len(h) -1)])
    h_rell = np.concatenate([h, np.zeros(len(x) -1)])
    Y = dft(x_rell) * dft(h_rell)
    # nos quedamos con la parte real
    return idft(Y).real
