"""Caso C, tarea C3: filtros notch 2D. Las mascaras se aplican sobre el espectro SIN shift."""

import numpy as np

from psi_lab.lab1.caso_imagen.fourier2d import dft2, idft2
from psi_lab.lab1.convolucion import convolucion_separable_2d


def _distancias_cuadradas(forma, pico):
    """Distancia^2 [bins^2] de cada bin (v, u) al `pico` (v0, u0), con distancia PERIODICA.

    El espectro de una imagen es periodico en ambos ejes, asi que la distancia de un bin a un
    pico se mide por el camino corto: min(|dv|, alto - |dv|) y lo mismo en columnas. Sin esto, un
    pico cerca del borde (p. ej. su conjugado, en alto - kv) quedaria con una mascara cortada.
    """
    alto, ancho = forma
    v0, u0 = int(pico[0]) % alto, int(pico[1]) % ancho
    dv = np.abs(np.arange(alto) - v0)
    du = np.abs(np.arange(ancho) - u0)
    dv = np.minimum(dv, alto - dv)[:, None]
    du = np.minimum(du, ancho - du)[None, :]
    return (dv**2 + du**2).astype(float)


def _como_picos(picos):
    picos = np.asarray(picos)
    if picos.size == 0:
        return picos.reshape(0, 2)
    if picos.ndim != 2 or picos.shape[1] != 2:
        raise ValueError(f"picos debe tener forma (n_picos, 2), no {picos.shape}")
    return picos


def notch_ideal(forma, picos, radio):
    """Mascara H float en [0, 1] de forma `forma`: 0 a distancia <= radio de cada pico, 1 fuera.

    picos: (n_picos, 2) enteros (fila, columna), con ambos picos de cada par conjugado.
    La distancia es periodica (el espectro de una imagen es periodico). radio en bins, >= 0.
    """
    if radio < 0:
        raise ValueError("radio debe ser >= 0")
    H = np.ones(tuple(forma), dtype=float)
    for pico in _como_picos(picos):
        H[_distancias_cuadradas(forma, pico) <= radio**2] = 0.0
    return H


def notch_gaussiano(forma, picos, sigma):
    """Mascara H float en [0, 1]: 1 - suma de gaussianas de desviacion sigma [px] en cada pico.

    Se usa el producto prod_i (1 - exp(-d_i^2 / 2 sigma^2)) en vez de la resta de la suma: es
    igual a ella cuando los notch no se traslapan, vale 0 exacto en cada pico y nunca es negativa.
    """
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    H = np.ones(tuple(forma), dtype=float)
    for pico in _como_picos(picos):
        H *= 1.0 - np.exp(-_distancias_cuadradas(forma, pico) / (2 * sigma**2))
    return H


def aplicar_mascara(img, H):
    """idft2(dft2(img) * H). Retorna complejo (la imagen filtrada es `.real`).

    Con un par conjugado completo en H la parte imaginaria es ~0.
    """
    img = np.asarray(img, dtype=float)
    H = np.asarray(H)
    if img.shape != H.shape:
        raise ValueError(f"img {img.shape} y H {H.shape} deben tener la misma forma")
    return idft2(dft2(img) * H)


def pasabajos_gaussiano(img, sigma):
    """Pasabajos espacial: convolucion de `img` con un nucleo gaussiano separable. Misma forma.

    Es la linea base contra la que se compara el notch. Nucleo de largo 2 ceil(3 sigma) + 1,
    normalizado a suma 1 (ganancia 1 en DC). Se extiende la imagen por reflejo antes de convolucionar
    para que los bordes no se oscurezcan (convolucion con ceros) y se recorta el centro despues.
    Usa `convolucion_separable_2d` del grupo; sigma en pixeles, > 0.
    """
    if sigma <= 0:
        raise ValueError("sigma debe ser > 0")
    img = np.asarray(img, dtype=float)
    r = int(np.ceil(3 * sigma))
    n = np.arange(-r, r + 1)
    h = np.exp(-(n**2) / (2 * sigma**2))
    h /= h.sum()
    alto, ancho = img.shape
    completa = convolucion_separable_2d(np.pad(img, r, mode="reflect"), h, h)
    return completa[2 * r : 2 * r + alto, 2 * r : 2 * r + ancho]
