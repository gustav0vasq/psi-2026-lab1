"""Caso C: metricas de calidad. SSIM se importa de skimage.metrics (structural_similarity)."""

import numpy as np


def psnr(a, b, rango=1.0):
    """PSNR [dB] = 10 log10(rango^2 / MSE(a, b)). Implementacion propia (lo exige el enunciado).

    a, b: arreglos de la misma forma; rango: valor maximo posible (1.0 para imagenes en [0, 1]).
    Retorna `inf` si a == b (MSE = 0).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"las formas no coinciden: {a.shape} y {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(rango**2 / mse))
