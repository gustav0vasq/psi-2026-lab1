"""Caso aplicado C: Fourier en 2D, ruido periodico y aliasing espacial.

Reexporta la API publica, asi se importa todo desde `psi_lab.lab1.caso_imagen`.
"""

from psi_lab.lab1.caso_imagen.aliasing import pasabajos_separable_2d, reducir, zone_plate
from psi_lab.lab1.caso_imagen.fourier2d import dft2, espectro_log, fase, idft2
from psi_lab.lab1.caso_imagen.metricas import psnr
from psi_lab.lab1.caso_imagen.notch import (
    aplicar_mascara,
    notch_gaussiano,
    notch_ideal,
    pasabajos_gaussiano,
)
from psi_lab.lab1.caso_imagen.ruido import agregar_interferencia, coseno_2d, detectar_picos

__all__ = [
    "agregar_interferencia",
    "aplicar_mascara",
    "coseno_2d",
    "detectar_picos",
    "dft2",
    "espectro_log",
    "fase",
    "idft2",
    "notch_gaussiano",
    "notch_ideal",
    "pasabajos_gaussiano",
    "pasabajos_separable_2d",
    "psnr",
    "reducir",
    "zone_plate",
]
