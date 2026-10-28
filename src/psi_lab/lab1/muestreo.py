"""Parte comun 1: muestreo y aliasing."""

import numpy as np

# Por implementar:

def muestrear(fun, fs, duracion):
    """Evalua fun(t) en t = n / fs, n = 0..N-1. Retorna (t, x)."""
    n = round(fs * duracion)
    t = np.arange(n) / fs
    return t, np.asarray(fun(t), dtype=float)

def frecuencia_alias(f, fs):
    """Frecuencia aparente en [0, fs/2] de una senoidal de f Hz muestreada a fs."""
    raise NotImplementedError