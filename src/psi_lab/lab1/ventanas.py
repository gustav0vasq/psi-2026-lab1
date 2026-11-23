"""Ventanas simetricas compartidas por el espectro (dft.py) y los filtros FIR (filtros.py)."""

import numpy as np

VENTANAS = ("rect", "hann", "hamming", "blackman")


def crear_ventana(nombre, n_coef):
    """Ventana simetrica de n_coef muestras: 'rect', 'hann', 'hamming' o 'blackman'."""
    if n_coef < 1:
        raise ValueError("n_coef debe ser >= 1")
    if nombre not in VENTANAS:
        raise ValueError(f"ventana desconocida: {nombre!r}; opciones: {VENTANAS}")
    if n_coef == 1 or nombre == "rect":
        return np.ones(n_coef)
    a = 2 * np.pi * np.arange(n_coef) / (n_coef - 1)
    if nombre == "hann":
        return 0.5 - 0.5 * np.cos(a)
    if nombre == "hamming":
        return 0.54 - 0.46 * np.cos(a)
    return 0.42 - 0.5 * np.cos(a) + 0.08 * np.cos(2 * a)  # blackman
