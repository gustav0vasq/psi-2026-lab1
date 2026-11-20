"""Tarea C3: filtros notch 2D (ideal y gaussiano), pasabajos espacial y metricas. Dueno: R3.

Auxiliares de biblioteca declarados (permitidos por el enunciado): `skimage.data.camera` (imagen),
`skimage.metrics.structural_similarity` (SSIM) y `scipy.ndimage.gaussian_filter` (pasabajos
espacial de comparacion). PSNR, DFT 2D, deteccion de picos y notch son propios.
"""

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage.metrics import structural_similarity

from psi_lab.lab1.caso_imagen import (
    agregar_interferencia,
    aplicar_mascara,
    detectar_picos,
    dft2,
    espectro_log,
    notch_gaussiano,
    notch_ideal,
    psnr,
)
from psi_lab.lab1.figuras import figura, guardar_tabla

# Interferencias (amplitud, ku, kv): ciclos enteros sobre la imagen de 512 x 512. Las dos primeras
# son las de la figura de R2; la tercera se suma en las pruebas con 3 interferencias.
INTERFERENCIAS = ((0.2, 40, 25), (0.12, 70, -50), (0.15, 100, 15))
UMBRAL = 0.3  # fraccion del maximo de |E| fuera del DC (ver ruido_sensibilidad, R2)
RADIO_ELEGIDO = 2
SIGMA_ELEGIDO = 2.0
SIGMA_PASABAJOS = 6.0  # px: el pasabajos hay que cerrarlo hasta bajar sobre la interferencia


def _imagen():
    """camera de skimage.data (viene con el paquete), float en [0, 1]."""
    return data.camera() / 255.0


def _con_ruido(img, n):
    for amplitud, ku, kv in INTERFERENCIAS[:n]:
        img = agregar_interferencia(img, ku, kv, amplitud)
    return img


def _metricas(limpia, restaurada):
    return psnr(restaurada, limpia), structural_similarity(limpia, restaurada, data_range=1.0)


def _filtrar(ruidosa, tipo, parametro):
    """Restaura `ruidosa` con el metodo `tipo`: detecta los picos solo y aplica la mascara."""
    if tipo == "pasabajos":
        return gaussian_filter(ruidosa, sigma=parametro, mode="reflect")
    picos = detectar_picos(dft2(ruidosa), UMBRAL)
    mascara = notch_ideal if tipo == "notch_ideal" else notch_gaussiano
    return aplicar_mascara(ruidosa, mascara(ruidosa.shape, picos, parametro)).real


@figura("notch_comparacion")
def _(salida):
    img = _imagen()
    ruidosa = _con_ruido(img, 2)
    E = dft2(ruidosa)
    picos = detectar_picos(E, UMBRAL)
    H_ideal = notch_ideal(img.shape, picos, RADIO_ELEGIDO)
    H_gauss = notch_gaussiano(img.shape, picos, SIGMA_ELEGIDO)
    resultados = [
        ("Con interferencia", ruidosa),
        (f"Notch ideal (r = {RADIO_ELEGIDO})", _filtrar(ruidosa, "notch_ideal", RADIO_ELEGIDO)),
        (f"Notch gaussiano (σ = {SIGMA_ELEGIDO:g})",
         _filtrar(ruidosa, "notch_gaussiano", SIGMA_ELEGIDO)),
        (f"Pasabajos gaussiano (σ = {SIGMA_PASABAJOS:g} px)",
         _filtrar(ruidosa, "pasabajos", SIGMA_PASABAJOS)),
    ]

    fig = Figure(figsize=(14, 7), layout="constrained")
    ejes = fig.subplots(2, 4)
    for ax, (titulo, im) in zip(ejes[0], resultados, strict=True):
        p, s = _metricas(img, im)
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{titulo}\nPSNR {p:.1f} dB, SSIM {s:.3f}", fontsize=9)
    espectros = [
        ("|E| con interferencia", espectro_log(E)),
        ("Máscara ideal (centrada)", np.fft.fftshift(H_ideal)),
        ("Máscara gaussiana (centrada)", np.fft.fftshift(H_gauss)),
        ("|E| tras el notch ideal", espectro_log(dft2(resultados[1][1]))),
    ]
    for ax, (titulo, im) in zip(ejes[1], espectros, strict=True):
        ax.imshow(im, cmap="gray")
        ax.set_title(titulo, fontsize=9)
    for ax in ejes.ravel():
        ax.axis("off")
    fig.savefig(salida / "notch_comparacion.png", dpi=120)


@figura("notch_parametros")
def _(salida):
    """PSNR y SSIM segun el radio / sigma / sigma del pasabajos, con 2 interferencias."""
    img = _imagen()
    ruidosa = _con_ruido(img, 2)
    barridos = [
        ("notch_ideal", "radio [bins]", (0, 1, 2, 4, 8, 16, 32)),
        ("notch_gaussiano", "σ [bins]", (0.5, 1, 2, 4, 8, 16)),
        ("pasabajos", "σ del pasabajos [px]", (1, 2, 4, 6, 8, 12)),
    ]
    fig = Figure(figsize=(12, 3.8), layout="constrained")
    for ax, (tipo, etiqueta, valores) in zip(fig.subplots(1, 3), barridos, strict=True):
        if tipo == "notch_ideal":
            valores = valores[1:]  # radio 0 borra solo el bin del pico: se muestra en la tabla
        medidas = [_metricas(img, _filtrar(ruidosa, tipo, v)) for v in valores]
        ax.plot(valores, [m[0] for m in medidas], "o-", label="PSNR [dB]")
        ax.axhline(psnr(ruidosa, img), color="gray", ls="--", label="sin filtrar")
        ax.set(xlabel=etiqueta, ylabel="PSNR [dB]", title=tipo.replace("_", " "))
        if tipo != "pasabajos":
            ax.set_xscale("log", base=2)
        ax2 = ax.twinx()
        ax2.plot(valores, [m[1] for m in medidas], "s--", color="C1")
        ax2.set_ylabel("SSIM", color="C1")
        ax.legend(fontsize=8, loc="lower left")
    fig.savefig(salida / "notch_parametros.png", dpi=120)


@figura("notch_tabla")
def _(salida):
    """Tabla PSNR/SSIM: metodo x parametro x numero de interferencias (1, 2, 3)."""
    img = _imagen()
    configuraciones = (
        [("notch_ideal", r) for r in (0, 1, 2, 4, 8, 16)]
        + [("notch_gaussiano", s) for s in (0.5, 1, 2, 4, 8)]
        + [("pasabajos", s) for s in (2, 4, 6, 8)]
    )
    filas = []
    for n in (1, 2, 3):
        ruidosa = _con_ruido(img, n)
        p0, s0 = _metricas(img, ruidosa)
        filas.append([n, "sin_filtrar", "", round(p0, 2), round(s0, 4)])
        for tipo, parametro in configuraciones:
            p, s = _metricas(img, _filtrar(ruidosa, tipo, parametro))
            filas.append([n, tipo, parametro, round(p, 2), round(s, 4)])
    guardar_tabla(salida, "notch_psnr_ssim",
                  ["interferencias", "metodo", "parametro", "psnr_db", "ssim"], filas)


def _con_ruido_fraccionario(img, ku, kv, amplitud):
    """Interferencia con ku, kv NO enteros: el periodo no calza con la imagen y el pico se abre."""
    alto, ancho = img.shape
    y, x = np.mgrid[0:alto, 0:ancho]
    return img + amplitud * np.cos(2 * np.pi * (ku * x / ancho + kv * y / alto))


@figura("notch_fuga")
def _(salida):
    """Por que el radio no puede ser 0 en una foto real: la fuga espectral ensancha el pico.

    Con ciclos enteros todo el pico cae en 1 bin y basta radio 0-1. Si la frecuencia cae entre
    dos bins (caso de un patron real), la fuga reparte energia en los bins vecinos y hay que
    cubrirla: se barre el radio para desfases de 0, 0.25 y 0.5 bin en ku.
    """
    img = _imagen()
    radios = (0, 1, 2, 3, 4, 6, 8, 12, 16)
    desfases = (0.0, 0.25, 0.5)
    filas = []
    fig = Figure(figsize=(6, 4), layout="constrained")
    ax = fig.subplots()
    for d in desfases:
        ruidosa = _con_ruido_fraccionario(img, 40 + d, 25, 0.2)
        picos = detectar_picos(dft2(ruidosa), UMBRAL)
        psnrs = []
        for r in radios:
            res = aplicar_mascara(ruidosa, notch_ideal(img.shape, picos, r)).real
            p, s = _metricas(img, res)
            psnrs.append(p)
            filas.append([d, r, len(picos), round(p, 2), round(s, 4)])
        ax.plot(radios, psnrs, "o-", label=f"desfase {d:g} bin")
    ax.set(xlabel="radio del notch ideal [bins]", ylabel="PSNR [dB]",
           title="Fuga espectral: el radio óptimo crece con el desfase")
    ax.legend(fontsize=8)
    fig.savefig(salida / "notch_fuga.png", dpi=120)
    guardar_tabla(salida, "notch_fuga", ["desfase_bins", "radio", "picos", "psnr_db", "ssim"], filas)
