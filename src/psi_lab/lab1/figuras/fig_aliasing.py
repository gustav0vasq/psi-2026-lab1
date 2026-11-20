"""Tarea C5: aliasing espacial con la zone plate y con una foto real"""

import numpy as np
from matplotlib.figure import Figure
from skimage import data

from psi_lab.lab1.caso_imagen import pasabajos_separable_2d, reducir, zone_plate
from psi_lab.lab1.figuras import figura

N_COEF = 51


@figura("aliasing_zone_plate")
def _(salida):
    z = zone_plate(512)
    fig = Figure(figsize=(12, 8))
    ejes = fig.subplots(2, 3)
    ejes[0, 0].imshow(z, cmap="gray", vmin=0, vmax=1)
    ejes[0, 0].set_title("Original 512 x 512")
    ejes[1, 0].axis("off")

    for col, k in enumerate([2, 4], start=1):
        fc = 0.5 / k - 0.04
        sin_filtro = reducir(z, k)
        con_filtro = reducir(pasabajos_separable_2d(z, fc, N_COEF), k)
        ejes[0, col].imshow(sin_filtro, cmap="gray", vmin=0, vmax=1)
        ejes[0, col].set_title(f"Reducida por {k}, sin filtro")
        ejes[1, col].imshow(con_filtro, cmap="gray", vmin=0, vmax=1)
        ejes[1, col].set_title(f"Reducida por {k}, pasabajos fc = {fc:.3f}")

    for ax in ejes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(salida / "aliasing_zone_plate.png", dpi=150)


@figura("aliasing_foto")
def _(salida):
    img = data.camera() / 255.0
    k = 4
    fc = 0.5 / k - 0.04
    sin_filtro = reducir(img, k)
    con_filtro = reducir(pasabajos_separable_2d(img, fc, N_COEF), k)

    fig = Figure(figsize=(12, 4.5))
    ejes = fig.subplots(1, 3)
    imagenes = [img, sin_filtro, con_filtro]
    titulos = [
        f"Original {img.shape[0]} x {img.shape[1]}",
        f"Reducida por {k}, sin filtro",
        f"Reducida por {k}, pasabajos fc = {fc:.3f}",
    ]
    for ax, im, titulo in zip(ejes, imagenes, titulos):
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(titulo)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(salida / "aliasing_foto.png", dpi=150)
