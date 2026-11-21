"""Tarea C4: notch sobre fotos con patron periodico REAL, sin imagen limpia de referencia.


Evaluacion sin referencia (no hay imagen limpia contra la cual medir PSNR/SSIM):
- fraccion del espectro que borra la mascara (mientras menor, mas "quirurgico"),
- energia en las zonas del notch antes y despues [dB],
- la imagen diferencia (original - filtrada) debe ser solo el patron: su correlacion con el
  contenido de baja frecuencia de la imagen debe ser ~0.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from skimage import color, io

from psi_lab.lab1 import datos
from psi_lab.lab1.caso_imagen import (
    aplicar_mascara,
    detectar_picos,
    dft2,
    espectro_log,
    idft2,
    notch_gaussiano,
)
from psi_lab.lab1.figuras import figura, guardar_tabla


@dataclass(frozen=True)
class Foto:
    nombre: str  # nombre corto, se usa en los archivos de salida
    archivo: str  # dentro de DIR_DATOS / "caso_real"
    recorte: tuple  # (slice filas, slice columnas): solo la zona con el patron, sin texto ni marcos
    umbral: float  # fraccion del maximo para detectar_picos
    radio_dc: int  # [bins] disco excluido alrededor del DC
    sigma: float  # [bins] ancho del notch gaussiano
    zoom: tuple  # (fila, columna, lado) de un cuadrado para ver el patron de cerca


FOTOS = (
    # Foto de diario (~1916) impresa con trama de puntos: el original no existe.
    Foto("diario", "diario.jpg", (slice(70, 920), slice(75, 655)), 0.1, 20, 4, (300, 150, 160)),
    # Foto propia de un monitor: la rejilla de pixeles de la pantalla queda en la foto.
    Foto("pantalla", "cielo.jpg", (slice(0, 1000), slice(None)), 0.3, 100, 2, (250, 1250, 120)),
)

RADIO_CONTENIDO = 0.05  # [ciclos/pixel] "contenido" = frecuencias por debajo de esto


def _cargar(foto):
    """Foto en escala de grises, float en [0, 1], recortada. None si el archivo no esta."""
    ruta = datos.DIR_DATOS / "caso_real" / foto.archivo
    if not ruta.exists():
        print(f"  [aviso] falta {ruta}: se salta la foto '{foto.nombre}'")
        return None
    img = io.imread(ruta)
    img = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img / 255.0
    return img.astype(float)[foto.recorte]


def _bajas_frecuencias(img, radio):
    """Contenido de la imagen: solo las frecuencias con |f| <= radio [ciclos/pixel]."""
    alto, ancho = img.shape
    fv = np.fft.fftfreq(alto)[:, None]
    fu = np.fft.fftfreq(ancho)[None, :]
    return idft2(dft2(img) * (np.hypot(fv, fu) <= radio)).real


def _evaluar(img, foto):
    E = dft2(img)
    picos = detectar_picos(E, foto.umbral, radio_dc=foto.radio_dc)
    H = notch_gaussiano(img.shape, picos, foto.sigma)
    filtrada = aplicar_mascara(img, H).real
    diferencia = img - filtrada

    zona = H < 0.5  # bins que el notch atenua a menos de la mitad
    energia_antes = np.sum(np.abs(E[zona]) ** 2)
    energia_despues = np.sum(np.abs(dft2(filtrada)[zona]) ** 2)
    contenido = _bajas_frecuencias(img, RADIO_CONTENIDO)
    metricas = {
        "picos": len(picos),
        "espectro_borrado_pct": 100 * (1 - H.mean()),
        "reduccion_energia_db": 10 * np.log10(energia_antes / max(energia_despues, 1e-300)),
        "corr_diferencia_contenido": np.corrcoef(diferencia.ravel(), contenido.ravel())[0, 1],
        "std_diferencia": diferencia.std(),
    }
    return E, picos, filtrada, diferencia, metricas


@figura("caso_real", requiere_datos=True)
def _(salida):
    filas = []
    for foto in FOTOS:
        img = _cargar(foto)
        if img is None:
            continue
        E, picos, filtrada, diferencia, m = _evaluar(img, foto)
        alto, ancho = img.shape

        fig = Figure(figsize=(16, 9), layout="constrained")
        arriba, abajo = fig.subplots(2, 4)
        f0, c0, lado = foto.zoom
        z = (slice(f0, f0 + lado), slice(c0, c0 + lado))
        paneles = (
            (img, "Original (sin referencia limpia)", {"vmin": 0, "vmax": 1}),
            (filtrada, f"Notch gaussiano, σ = {foto.sigma:g} bins", {"vmin": 0, "vmax": 1}),
            (diferencia, f"Diferencia (corr. con contenido = {m['corr_diferencia_contenido']:+.3f})",
             {}),
        )
        for ax_a, ax_b, (im, titulo, rango) in zip(arriba, abajo, paneles):
            ax_a.imshow(im, cmap="gray", **rango)
            ax_a.add_patch(Rectangle((c0, f0), lado, lado, fill=False, ec="red", lw=1))
            ax_a.set_title(titulo, fontsize=9)
            ax_b.imshow(im[z], cmap="gray", interpolation="nearest", **rango)
            ax_b.set_title("Zoom del recuadro rojo", fontsize=9)
        arriba[3].imshow(espectro_log(E), cmap="gray")
        arriba[3].scatter((picos[:, 1] + ancho // 2) % ancho, (picos[:, 0] + alto // 2) % alto,
                          s=12, facecolors="none", edgecolors="red", linewidths=0.8)
        arriba[3].set_title(f"Espectro: {m['picos']} bins detectados (umbral {foto.umbral:g})",
                            fontsize=9)
        for ax in (*arriba, *abajo):
            ax.axis("off")
        fig.savefig(salida / f"caso_real_{foto.nombre}.png", dpi=110)

        filas.append([foto.nombre, foto.umbral, foto.radio_dc, foto.sigma, m["picos"],
                      f"{m['espectro_borrado_pct']:.3f}", f"{m['reduccion_energia_db']:.1f}",
                      f"{m['corr_diferencia_contenido']:+.4f}", f"{m['std_diferencia']:.4f}"])
    if filas:
        guardar_tabla(salida, "caso_real",
                      ["foto", "umbral", "radio_dc", "sigma", "bins_detectados",
                       "espectro_borrado_pct", "reduccion_energia_db",
                       "corr_diferencia_contenido", "std_diferencia"], filas)