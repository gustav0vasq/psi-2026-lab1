"""Tarea C1: magnitud y fase de una imagen, y el experimento de intercambiar las fases."""

import numpy as np
from matplotlib.figure import Figure
from skimage import color, data

from psi_lab.lab1.caso_imagen import dft2, espectro_log, fase, idft2
from psi_lab.lab1.figuras import figura, guardar_tabla


def _imagen(nombre):
    """Imagen de skimage.data en escala de grises, float64 en [0, 1], 512 x 512."""
    img = {"camera": data.camera, "astronaut": data.astronaut}[nombre]()
    img = color.rgb2gray(img) if img.ndim == 3 else img / 255
    return img.astype(np.float64)


def _combinar(magnitud_de, fase_de):
    """Imagen con la magnitud del espectro de una imagen y la fase de otra."""
    E = np.abs(dft2(magnitud_de)) * np.exp(1j * fase(dft2(fase_de)))
    return idft2(E).real


@figura("fourier2d_espectro")
def _(salida):
    img = _imagen("camera")
    E = dft2(img)
    fig = Figure(figsize=(10, 3.4))
    paneles = (
        (img, "Imagen", "gray"),
        (espectro_log(E), "Magnitud, log(1 + |E|), centrada", "gray"),
        (np.fft.fftshift(fase(E)), "Fase [rad], centrada", "twilight"),
    )
    for ax, (im, titulo, cmap) in zip(fig.subplots(1, 3), paneles, strict=True):
        ax.imshow(im, cmap=cmap)
        ax.set_title(titulo, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(salida / "fourier2d_espectro.png", dpi=120)


@figura("fourier2d_fases")
def _(salida):
    a, b = _imagen("camera"), _imagen("astronaut")
    a_con_fase_de_b = _combinar(a, b)
    b_con_fase_de_a = _combinar(b, a)

    fig = Figure(figsize=(10, 5.4))
    paneles = (
        (a, "A: camera"),
        (b, "B: astronaut"),
        (a_con_fase_de_b, "magnitud de A + fase de B"),
        (b_con_fase_de_a, "magnitud de B + fase de A"),
    )
    for ax, (im, titulo) in zip(fig.subplots(2, 2).ravel(), paneles, strict=True):
        ax.imshow(im, cmap="gray")
        ax.set_title(titulo, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(salida / "fourier2d_fases.png", dpi=120)

    def corr(x, y):
        return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])

    guardar_tabla(
        salida,
        "fourier2d_fases",
        ["imagen", "corr_con_A", "corr_con_B"],
        [
            ["magnitud de A + fase de B", corr(a_con_fase_de_b, a), corr(a_con_fase_de_b, b)],
            ["magnitud de B + fase de A", corr(b_con_fase_de_a, a), corr(b_con_fase_de_a, b)],
        ],
    )
