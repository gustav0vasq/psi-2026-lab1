"""Caso C, tarea C2: interferencia periodica sintetica y deteccion de sus picos."""

import numpy as np


def coseno_2d(forma, ku, kv, amplitud):
    """Coseno amplitud * cos(2 pi (ku x / ancho + kv y / alto)), con forma = (alto, ancho).

    ku y kv son ciclos ENTEROS sobre la imagen completa: los picos caen exactamente en los
    bins (kv, ku) y (alto - kv, ancho - ku) del espectro sin shift.
    """
    alto, ancho = forma
    y = np.arange(alto)[:, None] # columna
    x = np.arange(ancho)[None, :]
    return amplitud * np.cos(2 * np.pi * (ku * x / ancho + kv * y / alto))


def agregar_interferencia(img, ku, kv, amplitud):
    """Suma coseno_2d(img.shape, ku, kv, amplitud) a img (sin recortar a [0, 1])."""
    img = np.asarray(img, dtype=float)
    return img + coseno_2d(img.shape, ku, kv, amplitud)

    
def detectar_picos(E, umbral, *, radio_dc = 10):
    """Picos de |E| por sobre `umbral` (fraccion del maximo de |E| fuera del DC).

    Excluye el DC (bin [0, 0]) y devuelve SIEMPRE ambos picos de cada par conjugado.
    Retorna un arreglo (n_picos, 2) de enteros, columnas (v, u) = (fila, columna).
    """
    mag = np.abs(np.asarray(E)).copy()
    alto, ancho = mag.shape

    # Excluir el disco del DC
    v = np.arange(alto)[:, None]
    u = np.arange(ancho)[None, :]
    distancia = np.hypot(np.minimum(v, alto - v), np.minimum(u, ancho - u))
    maximo_total = mag.max()
    mag[distancia <= radio_dc] = 0
    if mag.max() <= 1e-9 * maximo_total:
        return np.zeros((0, 2), dtype=int)

    # Bins que superan la fraccion umbral del maximo restante
    candidatos = np.argwhere(mag >= umbral * mag.max())

    # Agregar los conjugados (-v, -u) y quitar repetidos
    conjugados = np.column_stack([-candidatos[:, 0] % alto, -candidatos[:, 1] % ancho])
    return np.unique(np.vstack([candidatos, conjugados]), axis=0)

