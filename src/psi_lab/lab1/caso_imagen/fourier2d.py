"""Caso C, tarea C1: DFT 2D. El arreglo de trabajo va SIN fftshift (DC en E[0, 0])."""

import numpy as np


def _como_2d(a):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"se esperaba un arreglo 2D, no de {a.ndim} dimensiones")
    return a


def dft2(img):
    """DFT 2D como DFT 1D por filas y luego por columnas. Retorna E complejo, forma de img.

    E[v, u] = sum_y sum_x img[y, x] exp(-2j pi (v y / alto + u x / ancho)). Como el nucleo se
    separa en un producto, basta transformar cada fila (eje 1) y luego cada columna (eje 0).
    """
    img = _como_2d(img)
    return np.fft.fft(np.fft.fft(img, axis=1), axis=0)


def idft2(E):
    """DFT 2D inversa. Retorna un arreglo complejo; la imagen es `.real`."""
    E = _como_2d(E)
    return np.fft.ifft(np.fft.ifft(E, axis=1), axis=0)


def espectro_log(E):
    """Magnitud en escala log, YA centrada (fftshift). Solo para graficar.

    Retorna log(1 + |E|), con el DC en el centro (alto // 2, ancho // 2).
    """
    return np.log1p(np.abs(np.fft.fftshift(_como_2d(E))))


def fase(E):
    """Fase de E en radianes, en [-pi, pi]. Misma forma que E, sin shift."""
    return np.angle(_como_2d(E))
