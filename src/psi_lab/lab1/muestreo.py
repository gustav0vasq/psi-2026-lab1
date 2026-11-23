"""Parte comun 1: muestreo y aliasing."""

import numpy as np


def muestrear(fun, fs, duracion):
    """Evalua fun(t) en t = n / fs, n = 0..N-1. Retorna (t, x)."""
    n = round(fs * duracion)
    t = np.arange(n) / fs
    return t, np.asarray(fun(t), dtype=float)

def frecuencia_alias(f, fs):
    """Frecuencia aparente en [0, fs/2] de una senoidal de f Hz muestreada a fs.

    Las muestras de cos(2 pi f n / fs) no cambian si f se reemplaza por f + k fs (k entero,
    porque se suma 2 pi k n al angulo) ni por -f (el coseno es par). Todas las frecuencias
    f' = +-f + k fs son indistinguibles, y la representante en [0, fs/2] es la distancia de f
    al multiplo de fs mas cercano: |f - fs round(f / fs)|.
    """
    return np.abs(f - fs * np.round(f / fs))
